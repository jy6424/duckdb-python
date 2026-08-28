#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import ctypes
import glob
import io
import math
import mmap
import os
import queue
import threading
import time


EXCLUDED_AUTO_COLUMNS = set(["grid", "time", "levs"])


def new_timers():
    return {
        "dimension_read": 0.0,
        "fact_read": 0.0,
        "join": 0.0,
        "groupby_size": 0.0,
        "groupby_sum": 0.0,
        "groupby_count": 0.0,
        "count_from_rows": 0.0,
        "merge_concat": 0.0,
        "merge_groupby": 0.0,
        "row_count_sum": 0.0,
        "reader_wall": 0.0,
        "aggregate_wall": 0.0,
        "file_io_read": 0.0,
        "file_io_wall": 0.0,
        "decode_materialize": 0.0,
        "decode_wall": 0.0,
        "fact_rchar": 0,
        "fact_read_bytes": 0,
        "fact_syscr": 0,
        "dimension_rchar": 0,
        "dimension_read_bytes": 0,
        "dimension_syscr": 0,
        "materialized_bytes": 0,
        "reader_queue_put": 0.0,
        "io_uring_fallbacks": 0,
        "rmm_setup": 0.0,
    }


def add_elapsed(timers, name, start):
    timers[name] += time.time() - start


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir")
    parser.add_argument("--vars", default="qicps")
    parser.add_argument("--join-key", default="grid")
    parser.add_argument("--group-column", default="lats")
    parser.add_argument("--dimension-file", default="grid.parquet")
    parser.add_argument("--read-mode", default="per-file", choices=["per-file", "glob"])
    parser.add_argument("--reuse-dimension-mapping", action="store_true")
    parser.add_argument(
        "--infer-grid-from-row-order",
        action="store_true",
        help="skip fact grid reads and join by repeating the dimension group column in fact row order",
    )
    parser.add_argument("--assume-payload-all-valid", action="store_true")
    parser.add_argument(
        "--reader-threads",
        type=int,
        default=int(
            os.environ.get(
                "CUDF_READER_THREADS",
                os.environ.get("DUCKDB_GPU_PIPELINE_READER_THREADS", "1"),
            )
        ),
        help="number of Python reader workers used for cuDF read_parquet",
    )
    parser.add_argument(
        "--file-batch-size",
        type=int,
        default=None,
        help="number of Parquet files read by each cuDF reader task",
    )
    parser.add_argument(
        "--pipeline-queue-depth",
        type=int,
        default=int(os.environ.get("CUDF_PIPELINE_QUEUE_DEPTH", "1")),
        help="number of decoded cuDF fact batches allowed to wait between read and aggregate stages",
    )
    parser.add_argument(
        "--split-file-io",
        action="store_true",
        help="read Parquet file bytes in separate I/O workers before cuDF decode/materialization",
    )
    parser.add_argument(
        "--file-io-method",
        default=os.environ.get("CUDF_FILE_IO_METHOD", "read"),
        choices=["read", "io_uring"],
        help="file byte read method used by --split-file-io",
    )
    parser.add_argument(
        "--file-io-sqpoll",
        action="store_true",
        default=os.environ.get("CUDF_FILE_IO_SQPOLL", "0") == "1",
        help="enable IORING_SETUP_SQPOLL for --file-io-method io_uring",
    )
    parser.add_argument(
        "--file-io-sqpoll-idle-ms",
        type=int,
        default=int(os.environ.get("CUDF_FILE_IO_SQPOLL_IDLE_MS", "2000")),
        help="SQPOLL kernel thread idle timeout in milliseconds",
    )
    parser.add_argument(
        "--io-queue-depth",
        type=int,
        default=int(os.environ.get("CUDF_IO_QUEUE_DEPTH", "1")),
        help="number of raw Parquet file-byte batches allowed to wait before decode",
    )
    parser.add_argument(
        "--decode-threads",
        type=int,
        default=int(os.environ.get("CUDF_DECODE_THREADS", "1")),
        help="number of cuDF decode/materialize workers used with --split-file-io",
    )
    parser.add_argument(
        "--rmm-pool",
        action="store_true",
        default=False,
        help="enable RMM pool allocator before importing cuDF",
    )
    parser.add_argument(
        "--rmm-pool-size",
        default=os.environ.get("CUDF_RMM_POOL_SIZE", "4GB"),
        help="initial RMM pool size, e.g. 2GB, 4096MB, or 4294967296",
    )
    parser.add_argument("--print-stage-times", action="store_true")
    parser.add_argument("--print-batch-times", action="store_true")
    parser.add_argument("--print-results", action="store_true")
    return parser.parse_args()


def read_proc_io():
    data = {}
    try:
        with open("/proc/self/io", "r") as handle:
            for line in handle:
                key, value = line.strip().split(":")
                data[key.strip()] = int(value.strip())
    except OSError:
        pass
    return data


def io_diff(before, after, key):
    return after.get(key, 0) - before.get(key, 0)


def add_io_diff(timers, prefix, before, after):
    timers[prefix + "_rchar"] += io_diff(before, after, "rchar")
    timers[prefix + "_read_bytes"] += io_diff(before, after, "read_bytes")
    timers[prefix + "_syscr"] += io_diff(before, after, "syscr")


def dataframe_nbytes(frame):
    try:
        usage = frame.memory_usage(deep=True)
        return int(usage.sum())
    except Exception:
        pass
    total = 0
    for column in getattr(frame, "columns", []):
        try:
            total += int(frame[column].memory_usage(deep=True))
        except Exception:
            pass
    return total


def parse_byte_size(value):
    text = str(value).strip().lower()
    if not text:
        raise ValueError("empty byte size")

    units = [
        ("gib", 1024 ** 3),
        ("gb", 1024 ** 3),
        ("mib", 1024 ** 2),
        ("mb", 1024 ** 2),
        ("kib", 1024),
        ("kb", 1024),
        ("b", 1),
    ]
    for suffix, multiplier in units:
        if text.endswith(suffix):
            return int(float(text[:-len(suffix)]) * multiplier)
    return int(text)


def configure_rmm_pool(args, timers):
    if not args.rmm_pool:
        return False, 0

    timer_start = time.time()
    pool_size = parse_byte_size(args.rmm_pool_size)
    try:
        import rmm

        rmm.reinitialize(pool_allocator=True, initial_pool_size=pool_size)
        try:
            import cupy

            allocator = getattr(rmm, "rmm_cupy_allocator", None)
            if allocator is None:
                allocator = rmm.allocators.cupy.rmm_cupy_allocator
            cupy.cuda.set_allocator(allocator)
        except Exception:
            pass
    except Exception as exc:
        raise SystemExit("failed to initialize RMM pool allocator: {}".format(exc))
    finally:
        add_elapsed(timers, "rmm_setup", timer_start)
    return True, pool_size


def read_auto_payload_columns(parquet_path):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required for --vars all metadata inspection: {}".format(exc))

    schema = pq.ParquetFile(parquet_path).schema_arrow
    return [
        field.name
        for field in schema
        if field.name not in EXCLUDED_AUTO_COLUMNS and str(field.type).lower() == "double"
    ]


def resolve_payload_columns(args, parquet_paths):
    if args.vars.strip().lower() == "all":
        return read_auto_payload_columns(parquet_paths[0])
    return [column.strip() for column in args.vars.split(",") if column.strip()]


def resolve_dimension_path(fact_path, dimension_file):
    if os.path.isabs(dimension_file) or os.path.dirname(dimension_file):
        return dimension_file
    return os.path.join(os.path.dirname(fact_path), dimension_file)


def make_counts_from_row_counts(row_counts, group_column, payload_columns):
    counts = row_counts[[group_column]].copy()
    for payload_column in payload_columns:
        counts[payload_column] = row_counts["row_count"]
    return counts


def aggregate_joined(
    joined,
    group_column,
    payload_columns,
    assume_payload_all_valid,
    timers,
    need_counts=True,
    need_row_counts=True,
):
    grouped = joined.groupby(group_column)
    row_counts = None
    if need_row_counts or (need_counts and assume_payload_all_valid):
        timer_start = time.time()
        row_counts = grouped.size().reset_index()
        row_counts = row_counts.rename(columns={row_counts.columns[-1]: "row_count"})
        cuda_synchronize()
        add_elapsed(timers, "groupby_size", timer_start)

    timer_start = time.time()
    sums = grouped[payload_columns].sum().reset_index()
    cuda_synchronize()
    add_elapsed(timers, "groupby_sum", timer_start)

    counts = None
    if not need_counts:
        return sums, counts, row_counts

    if assume_payload_all_valid:
        timer_start = time.time()
        counts = make_counts_from_row_counts(row_counts, group_column, payload_columns)
        cuda_synchronize()
        add_elapsed(timers, "count_from_rows", timer_start)
    else:
        timer_start = time.time()
        counts = grouped[payload_columns].count().reset_index()
        cuda_synchronize()
        add_elapsed(timers, "groupby_count", timer_start)
    return sums, counts, row_counts


def append_frame(frames, frame):
    if frame is not None and len(frame) > 0:
        frames.append(frame)


def cuda_synchronize():
    try:
        import cupy

        cupy.cuda.Stream.null.synchronize()
        return
    except Exception:
        pass
    try:
        from numba import cuda

        cuda.synchronize()
    except Exception:
        pass


def read_parquet_gpu(cudf, source, columns):
    return cudf.read_parquet(source, columns=columns)


def timed_read_parquet_gpu(cudf, source, columns, measure_materialized_bytes=False):
    io_before = read_proc_io()
    timer_start = time.time()
    frame = read_parquet_gpu(cudf, source, columns=columns)
    cuda_synchronize()
    elapsed = time.time() - timer_start
    io_after = read_proc_io()
    profile = {
        "elapsed": elapsed,
        "rchar": io_diff(io_before, io_after, "rchar"),
        "read_bytes": io_diff(io_before, io_after, "read_bytes"),
        "syscr": io_diff(io_before, io_after, "syscr"),
        "rows": len(frame),
        "materialized_bytes": 0,
    }
    if measure_materialized_bytes:
        profile["materialized_bytes"] = dataframe_nbytes(frame)
    return frame, profile


class IoSqringOffsets(ctypes.Structure):
    _fields_ = [
        ("head", ctypes.c_uint32),
        ("tail", ctypes.c_uint32),
        ("ring_mask", ctypes.c_uint32),
        ("ring_entries", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("dropped", ctypes.c_uint32),
        ("array", ctypes.c_uint32),
        ("resv1", ctypes.c_uint32),
        ("user_addr", ctypes.c_uint64),
    ]


class IoCqringOffsets(ctypes.Structure):
    _fields_ = [
        ("head", ctypes.c_uint32),
        ("tail", ctypes.c_uint32),
        ("ring_mask", ctypes.c_uint32),
        ("ring_entries", ctypes.c_uint32),
        ("overflow", ctypes.c_uint32),
        ("cqes", ctypes.c_uint32),
        ("flags", ctypes.c_uint64),
        ("resv1", ctypes.c_uint64),
    ]


class IoUringParams(ctypes.Structure):
    _fields_ = [
        ("sq_entries", ctypes.c_uint32),
        ("cq_entries", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("sq_thread_cpu", ctypes.c_uint32),
        ("sq_thread_idle", ctypes.c_uint32),
        ("features", ctypes.c_uint32),
        ("wq_fd", ctypes.c_uint32),
        ("resv", ctypes.c_uint32 * 3),
        ("sq_off", IoSqringOffsets),
        ("cq_off", IoCqringOffsets),
    ]


class IoUringSqe(ctypes.Structure):
    _fields_ = [
        ("opcode", ctypes.c_uint8),
        ("flags", ctypes.c_uint8),
        ("ioprio", ctypes.c_uint16),
        ("fd", ctypes.c_int32),
        ("off", ctypes.c_uint64),
        ("addr", ctypes.c_uint64),
        ("len", ctypes.c_uint32),
        ("rw_flags", ctypes.c_uint32),
        ("user_data", ctypes.c_uint64),
        ("buf_index", ctypes.c_uint16),
        ("personality", ctypes.c_uint16),
        ("splice_fd_in", ctypes.c_int32),
        ("pad2", ctypes.c_uint64 * 2),
    ]


class IoUringCqe(ctypes.Structure):
    _fields_ = [
        ("user_data", ctypes.c_uint64),
        ("res", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
    ]


class IOVec(ctypes.Structure):
    _fields_ = [
        ("iov_base", ctypes.c_void_p),
        ("iov_len", ctypes.c_size_t),
    ]


class SimpleIoUring(object):
    SYS_IO_URING_SETUP = 425
    SYS_IO_URING_ENTER = 426
    SYS_IO_URING_REGISTER = 427
    IORING_OFF_SQ_RING = 0
    IORING_OFF_CQ_RING = 0x8000000
    IORING_OFF_SQES = 0x10000000
    IORING_ENTER_GETEVENTS = 1
    IORING_ENTER_SQ_WAKEUP = 2
    IORING_SETUP_SQPOLL = 2
    IORING_SQ_NEED_WAKEUP = 1
    IORING_OP_READV = 1
    IORING_REGISTER_FILES = 2
    IORING_UNREGISTER_FILES = 3
    IOSQE_FIXED_FILE = 1

    def __init__(self, entries, sqpoll=False, sqpoll_idle_ms=2000):
        if not hasattr(os, "uname") or os.uname().sysname != "Linux":
            raise OSError("io_uring is only available on Linux")
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.entries = max(1, int(entries))
        self.sqpoll = bool(sqpoll)
        self.fixed_files_registered = False
        self.params = IoUringParams()
        if self.sqpoll:
            self.params.flags |= self.IORING_SETUP_SQPOLL
            self.params.sq_thread_idle = max(1, int(sqpoll_idle_ms))
        fd = self.libc.syscall(self.SYS_IO_URING_SETUP, self.entries, ctypes.byref(self.params))
        if fd < 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))
        self.ring_fd = int(fd)
        self.sq_ring = None
        self.cq_ring = None
        self.sqes_map = None
        self._map_rings()

    def _map_rings(self):
        sq_ring_size = self.params.sq_off.array + self.params.sq_entries * ctypes.sizeof(ctypes.c_uint32)
        cq_ring_size = self.params.cq_off.cqes + self.params.cq_entries * ctypes.sizeof(IoUringCqe)
        sqes_size = self.params.sq_entries * ctypes.sizeof(IoUringSqe)
        self.sq_ring = mmap.mmap(
            self.ring_fd,
            sq_ring_size,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
            offset=self.IORING_OFF_SQ_RING,
        )
        self.cq_ring = mmap.mmap(
            self.ring_fd,
            cq_ring_size,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
            offset=self.IORING_OFF_CQ_RING,
        )
        self.sqes_map = mmap.mmap(
            self.ring_fd,
            sqes_size,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
            offset=self.IORING_OFF_SQES,
        )
        self.sq_head = ctypes.c_uint32.from_buffer(self.sq_ring, self.params.sq_off.head)
        self.sq_tail = ctypes.c_uint32.from_buffer(self.sq_ring, self.params.sq_off.tail)
        self.sq_mask = ctypes.c_uint32.from_buffer(self.sq_ring, self.params.sq_off.ring_mask)
        self.sq_flags = ctypes.c_uint32.from_buffer(self.sq_ring, self.params.sq_off.flags)
        self.sq_array = (ctypes.c_uint32 * self.params.sq_entries).from_buffer(
            self.sq_ring,
            self.params.sq_off.array,
        )
        self.cq_head = ctypes.c_uint32.from_buffer(self.cq_ring, self.params.cq_off.head)
        self.cq_tail = ctypes.c_uint32.from_buffer(self.cq_ring, self.params.cq_off.tail)
        self.cq_mask = ctypes.c_uint32.from_buffer(self.cq_ring, self.params.cq_off.ring_mask)
        self.cqes = (IoUringCqe * self.params.cq_entries).from_buffer(
            self.cq_ring,
            self.params.cq_off.cqes,
        )
        self.sqes = (IoUringSqe * self.params.sq_entries).from_buffer(self.sqes_map, 0)

    def register_files(self, fds):
        if not fds:
            return
        fd_array = (ctypes.c_int * len(fds))(*fds)
        result = self.libc.syscall(
            self.SYS_IO_URING_REGISTER,
            self.ring_fd,
            self.IORING_REGISTER_FILES,
            ctypes.byref(fd_array),
            len(fds),
        )
        if result < 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))
        self.fixed_files_registered = True

    def submit_readv(self, fd, iovec, size, user_data, fixed_file=False):
        head = self.sq_head.value
        tail = self.sq_tail.value
        if tail - head >= self.params.sq_entries:
            raise RuntimeError("io_uring submission queue is full")
        index = tail & self.sq_mask.value
        sqe = self.sqes[index]
        ctypes.memset(ctypes.addressof(sqe), 0, ctypes.sizeof(IoUringSqe))
        sqe.opcode = self.IORING_OP_READV
        sqe.fd = int(fd)
        if fixed_file:
            sqe.flags |= self.IOSQE_FIXED_FILE
        sqe.off = 0
        sqe.addr = ctypes.addressof(iovec)
        sqe.len = 1
        sqe.user_data = int(user_data)
        self.sq_array[index] = index
        self.sq_tail.value = tail + 1

    def wake_sqpoll_if_needed(self):
        if not self.sqpoll:
            return
        if self.sq_flags.value & self.IORING_SQ_NEED_WAKEUP:
            result = self.libc.syscall(
                self.SYS_IO_URING_ENTER,
                self.ring_fd,
                0,
                0,
                self.IORING_ENTER_SQ_WAKEUP,
                None,
                0,
            )
            if result < 0:
                err = ctypes.get_errno()
                raise OSError(err, os.strerror(err))

    def enter(self, submit_count, wait_count):
        result = self.libc.syscall(
            self.SYS_IO_URING_ENTER,
            self.ring_fd,
            int(submit_count),
            int(wait_count),
            self.IORING_ENTER_GETEVENTS,
            None,
            0,
        )
        if result < 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))
        return int(result)

    def collect(self, count):
        completed = {}
        submitted = count
        while len(completed) < count:
            if self.sqpoll:
                self.wake_sqpoll_if_needed()
                self.enter(0, 1)
            else:
                self.enter(submitted, 1)
            submitted = 0
            while self.cq_head.value != self.cq_tail.value:
                head = self.cq_head.value
                cqe = self.cqes[head & self.cq_mask.value]
                completed[int(cqe.user_data)] = int(cqe.res)
                self.cq_head.value = head + 1
        return completed

    def close(self):
        if getattr(self, "ring_fd", -1) >= 0:
            if self.fixed_files_registered:
                self.libc.syscall(
                    self.SYS_IO_URING_REGISTER,
                    self.ring_fd,
                    self.IORING_UNREGISTER_FILES,
                    None,
                    0,
                )
                self.fixed_files_registered = False
            os.close(self.ring_fd)
            self.ring_fd = -1


def read_file_bytes_blocking(paths):
    buffers = []
    byte_count = 0
    for path in paths:
        with open(path, "rb") as handle:
            data = handle.read()
        byte_count += len(data)
        buffers.append(data)
    return buffers, byte_count


def read_file_bytes_io_uring(paths, sqpoll=False, sqpoll_idle_ms=2000):
    ring = SimpleIoUring(len(paths), sqpoll=sqpoll, sqpoll_idle_ms=sqpoll_idle_ms)
    fds = []
    raw_buffers = []
    iovecs = []
    sizes = []
    try:
        for path in paths:
            fd = os.open(path, os.O_RDONLY)
            fds.append(fd)
        if sqpoll:
            ring.register_files(fds)
        for idx, fd in enumerate(fds):
            size = os.fstat(fd).st_size
            sizes.append(size)
            buffer = ctypes.create_string_buffer(size)
            iovec = IOVec(ctypes.cast(buffer, ctypes.c_void_p), size)
            raw_buffers.append(buffer)
            iovecs.append(iovec)
            submit_fd = idx if sqpoll else fd
            ring.submit_readv(submit_fd, iovec, size, idx, fixed_file=sqpoll)
        results = ring.collect(len(paths))
        for idx, size in enumerate(sizes):
            res = results.get(idx, -1)
            if res < 0:
                errno_value = -res
                raise OSError(errno_value, os.strerror(errno_value))
            if res != size:
                raise OSError("short io_uring read for {}: {} of {}".format(paths[idx], res, size))
        buffers = [raw_buffers[idx].raw for idx in range(len(raw_buffers))]
        return buffers, sum(sizes)
    finally:
        try:
            ring.close()
        except Exception:
            pass
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass


def timed_read_file_bytes(paths, method="read", sqpoll=False, sqpoll_idle_ms=2000):
    io_before = read_proc_io()
    timer_start = time.time()
    actual_method = method
    fallback_reason = ""
    if method == "io_uring":
        try:
            buffers, byte_count = read_file_bytes_io_uring(paths, sqpoll=sqpoll, sqpoll_idle_ms=sqpoll_idle_ms)
        except Exception as exc:
            actual_method = "read-fallback"
            fallback_reason = str(exc)
            buffers, byte_count = read_file_bytes_blocking(paths)
    else:
        buffers, byte_count = read_file_bytes_blocking(paths)
    elapsed = time.time() - timer_start
    io_after = read_proc_io()
    return buffers, {
        "elapsed": elapsed,
        "file_bytes": byte_count,
        "method": actual_method + ("+sqpoll" if actual_method == "io_uring" and sqpoll else ""),
        "fallback_reason": fallback_reason,
        "rchar": io_diff(io_before, io_after, "rchar"),
        "read_bytes": io_diff(io_before, io_after, "read_bytes"),
        "syscr": io_diff(io_before, io_after, "syscr"),
    }


def timed_decode_parquet_bytes(cudf, buffers, columns, measure_materialized_bytes=False):
    sources = [io.BytesIO(data) for data in buffers]
    source = sources[0] if len(sources) == 1 else sources
    timer_start = time.time()
    frame = read_parquet_gpu(cudf, source, columns=columns)
    cuda_synchronize()
    elapsed = time.time() - timer_start
    profile = {
        "elapsed": elapsed,
        "rows": len(frame),
        "materialized_bytes": 0,
    }
    if measure_materialized_bytes:
        profile["materialized_bytes"] = dataframe_nbytes(frame)
    return frame, profile


def make_file_batches(paths, batch_size):
    return [paths[idx:idx + batch_size] for idx in range(0, len(paths), batch_size)]


def default_file_batch_size(read_mode, reader_threads, file_count):
    if read_mode == "glob":
        return int(math.ceil(float(file_count) / float(reader_threads)))
    return 1


def resolve_file_batch_size(args, reader_threads, file_count):
    if args.file_batch_size is not None:
        file_batch_size = args.file_batch_size
    else:
        file_batch_size = default_file_batch_size(args.read_mode, reader_threads, file_count)
    if file_batch_size <= 0:
        raise ValueError("--file-batch-size must be positive")
    return file_batch_size


def attach_group_from_row_order(cudf, fact, dim, group_column):
    dim_rows = len(dim)
    fact_rows = len(fact)
    if dim_rows <= 0:
        raise ValueError("dimension file has no rows")
    if fact_rows % dim_rows != 0:
        raise ValueError(
            "cannot infer {} from row order: fact rows {} is not a multiple of dimension rows {}".format(
                group_column, fact_rows, dim_rows
            )
        )

    repeats = fact_rows // dim_rows
    group_values = dim[group_column]
    if repeats == 1:
        fact[group_column] = group_values
    else:
        try:
            import cupy

            fact[group_column] = cudf.Series(cupy.tile(group_values.values, repeats))
        except Exception:
            fact[group_column] = cudf.concat([group_values] * repeats, ignore_index=True)
    return fact


def threaded_read_aggregate(
    cudf,
    args,
    parquet_paths,
    payload_columns,
    detail_timers,
    sum_frames,
    count_frames,
    row_count_frames,
    batch_profiles,
):
    reader_threads = max(1, min(args.reader_threads, len(parquet_paths)))
    file_batch_size = resolve_file_batch_size(args, reader_threads, len(parquet_paths))
    queue_depth = max(1, args.pipeline_queue_depth)

    fact_columns = payload_columns if args.infer_grid_from_row_order else [args.join_key] + payload_columns
    dimension_columns = [args.group_column] if args.infer_grid_from_row_order else [args.join_key, args.group_column]
    need_counts = args.print_results
    need_row_counts = args.print_results

    shared_dim = None
    if args.reuse_dimension_mapping or args.infer_grid_from_row_order:
        dimension_path = resolve_dimension_path(parquet_paths[0], args.dimension_file)
        shared_dim, dim_profile = timed_read_parquet_gpu(
            cudf,
            dimension_path,
            columns=dimension_columns,
            measure_materialized_bytes=args.print_stage_times or args.print_batch_times,
        )
        detail_timers["dimension_read"] += dim_profile["elapsed"]
        detail_timers["dimension_rchar"] += dim_profile["rchar"]
        detail_timers["dimension_read_bytes"] += dim_profile["read_bytes"]
        detail_timers["dimension_syscr"] += dim_profile["syscr"]

    tasks = queue.Queue()
    for batch_id, batch_paths in enumerate(make_file_batches(parquet_paths, file_batch_size)):
        tasks.put((batch_id, batch_paths))

    results = queue.Queue(maxsize=queue_depth)
    worker_wall_times = []

    def reader_worker():
        worker_start = time.time()
        try:
            while True:
                try:
                    batch_id, batch_paths = tasks.get_nowait()
                except queue.Empty:
                    break

                source = batch_paths[0] if len(batch_paths) == 1 else batch_paths
                fact, fact_profile = timed_read_parquet_gpu(
                    cudf,
                    source,
                    columns=fact_columns,
                    measure_materialized_bytes=args.print_stage_times or args.print_batch_times,
                )
                put_start = time.time()
                results.put(("batch", batch_id, batch_paths, fact, fact_profile))
                detail_timers["reader_queue_put"] += time.time() - put_start
        except BaseException as exc:
            results.put(("error", exc))
        finally:
            worker_wall_times.append(time.time() - worker_start)
            results.put(("done",))

    workers = [threading.Thread(target=reader_worker) for _ in range(reader_threads)]
    for worker in workers:
        worker.start()

    done_workers = 0
    read_seconds = 0.0
    join_seconds = 0.0
    group_seconds = 0.0
    row_count = 0

    while done_workers < reader_threads:
        item = results.get()
        item_type = item[0]
        if item_type == "done":
            done_workers += 1
            continue
        if item_type == "error":
            for worker in workers:
                worker.join()
            raise item[1]

        _, batch_id, batch_paths, fact, fact_profile = item
        detail_timers["fact_read"] += fact_profile["elapsed"]
        detail_timers["fact_rchar"] += fact_profile["rchar"]
        detail_timers["fact_read_bytes"] += fact_profile["read_bytes"]
        detail_timers["fact_syscr"] += fact_profile["syscr"]
        detail_timers["materialized_bytes"] += fact_profile["materialized_bytes"]
        batch_profile = {
            "batch_id": batch_id,
            "files": len(batch_paths),
            "read": fact_profile["elapsed"],
            "file_io": 0.0,
            "decode": fact_profile["elapsed"],
            "io_method": "cudf",
            "io_fallback_reason": "",
            "rows": fact_profile["rows"],
            "rchar": fact_profile["rchar"],
            "read_bytes": fact_profile["read_bytes"],
            "syscr": fact_profile["syscr"],
            "materialized_bytes": fact_profile["materialized_bytes"],
            "join": 0.0,
            "groupby": 0.0,
        }

        if shared_dim is None:
            dimension_path = resolve_dimension_path(batch_paths[0], args.dimension_file)
            dim, dim_profile = timed_read_parquet_gpu(
                cudf,
                dimension_path,
                columns=dimension_columns,
                measure_materialized_bytes=args.print_stage_times or args.print_batch_times,
            )
            detail_timers["dimension_read"] += dim_profile["elapsed"]
            detail_timers["dimension_rchar"] += dim_profile["rchar"]
            detail_timers["dimension_read_bytes"] += dim_profile["read_bytes"]
            detail_timers["dimension_syscr"] += dim_profile["syscr"]
        else:
            dim = shared_dim

        join_start = time.time()
        if args.infer_grid_from_row_order:
            joined = attach_group_from_row_order(cudf, fact, dim, args.group_column)
        else:
            joined = fact.merge(dim, on=args.join_key, how="inner")
        cuda_synchronize()
        join_elapsed = time.time() - join_start
        join_seconds += join_elapsed
        detail_timers["join"] += join_elapsed
        batch_profile["join"] = join_elapsed
        row_count += len(joined)

        group_start = time.time()
        sums, counts, row_counts = aggregate_joined(
            joined,
            args.group_column,
            payload_columns,
            args.assume_payload_all_valid,
            detail_timers,
            need_counts=need_counts,
            need_row_counts=need_row_counts,
        )
        cuda_synchronize()
        group_elapsed = time.time() - group_start
        group_seconds += group_elapsed
        detail_timers["aggregate_wall"] += group_elapsed
        batch_profile["groupby"] = group_elapsed
        append_frame(sum_frames, sums)
        append_frame(count_frames, counts)
        append_frame(row_count_frames, row_counts)
        batch_profiles.append(batch_profile)
        del fact
        del joined

    for worker in workers:
        worker.join()

    detail_timers["reader_wall"] = max(worker_wall_times) if worker_wall_times else 0.0
    read_seconds = detail_timers["dimension_read"] + detail_timers["reader_wall"]
    return read_seconds, join_seconds, group_seconds, row_count


def split_file_io_read_aggregate(
    cudf,
    args,
    parquet_paths,
    payload_columns,
    detail_timers,
    sum_frames,
    count_frames,
    row_count_frames,
    batch_profiles,
):
    io_threads = max(1, min(args.reader_threads, len(parquet_paths)))
    decode_threads = max(1, args.decode_threads)
    file_batch_size = resolve_file_batch_size(args, io_threads, len(parquet_paths))
    io_queue_depth = max(1, args.io_queue_depth)
    pipeline_queue_depth = max(1, args.pipeline_queue_depth)

    fact_columns = payload_columns if args.infer_grid_from_row_order else [args.join_key] + payload_columns
    dimension_columns = [args.group_column] if args.infer_grid_from_row_order else [args.join_key, args.group_column]
    need_counts = args.print_results
    need_row_counts = args.print_results
    measure_materialized_bytes = args.print_stage_times or args.print_batch_times

    shared_dim = None
    if args.reuse_dimension_mapping or args.infer_grid_from_row_order:
        dimension_path = resolve_dimension_path(parquet_paths[0], args.dimension_file)
        shared_dim, dim_profile = timed_read_parquet_gpu(
            cudf,
            dimension_path,
            columns=dimension_columns,
            measure_materialized_bytes=measure_materialized_bytes,
        )
        detail_timers["dimension_read"] += dim_profile["elapsed"]
        detail_timers["dimension_rchar"] += dim_profile["rchar"]
        detail_timers["dimension_read_bytes"] += dim_profile["read_bytes"]
        detail_timers["dimension_syscr"] += dim_profile["syscr"]

    tasks = queue.Queue()
    for batch_id, batch_paths in enumerate(make_file_batches(parquet_paths, file_batch_size)):
        tasks.put((batch_id, batch_paths))

    raw_batches = queue.Queue(maxsize=io_queue_depth)
    decoded_batches = queue.Queue(maxsize=pipeline_queue_depth)
    io_wall_times = []
    decode_wall_times = []
    timer_lock = threading.Lock()

    def add_counter(name, value):
        with timer_lock:
            detail_timers[name] += value

    def io_worker():
        worker_start = time.time()
        try:
            while True:
                try:
                    batch_id, batch_paths = tasks.get_nowait()
                except queue.Empty:
                    break

                buffers, io_profile = timed_read_file_bytes(
                    batch_paths,
                    method=args.file_io_method,
                    sqpoll=args.file_io_sqpoll,
                    sqpoll_idle_ms=args.file_io_sqpoll_idle_ms,
                )
                add_counter("file_io_read", io_profile["elapsed"])
                add_counter("fact_read_bytes", io_profile["read_bytes"])
                add_counter("fact_rchar", io_profile["rchar"])
                add_counter("fact_syscr", io_profile["syscr"])
                if args.file_io_method == "io_uring" and not io_profile["method"].startswith("io_uring"):
                    add_counter("io_uring_fallbacks", 1)
                raw_batches.put(("raw", batch_id, batch_paths, buffers, io_profile))
        except BaseException as exc:
            decoded_batches.put(("error", exc))
        finally:
            io_wall_times.append(time.time() - worker_start)

    def decode_worker():
        worker_start = time.time()
        try:
            while True:
                item = raw_batches.get()
                if item[0] == "done":
                    break
                _, batch_id, batch_paths, buffers, io_profile = item

                fact, decode_profile = timed_decode_parquet_bytes(
                    cudf,
                    buffers,
                    columns=fact_columns,
                    measure_materialized_bytes=measure_materialized_bytes,
                )
                add_counter("decode_materialize", decode_profile["elapsed"])
                add_counter("fact_read", decode_profile["elapsed"])
                add_counter("materialized_bytes", decode_profile["materialized_bytes"])
                batch_profile = {
                    "batch_id": batch_id,
                    "files": len(batch_paths),
                    "read": io_profile["elapsed"] + decode_profile["elapsed"],
                    "file_io": io_profile["elapsed"],
                    "decode": decode_profile["elapsed"],
                    "io_method": io_profile["method"],
                    "io_fallback_reason": io_profile["fallback_reason"],
                    "rows": decode_profile["rows"],
                    "rchar": io_profile["rchar"],
                    "read_bytes": io_profile["read_bytes"],
                    "syscr": io_profile["syscr"],
                    "materialized_bytes": decode_profile["materialized_bytes"],
                    "join": 0.0,
                    "groupby": 0.0,
                }
                del buffers
                decoded_batches.put(("batch", batch_id, batch_paths, fact, batch_profile))
        except BaseException as exc:
            decoded_batches.put(("error", exc))
        finally:
            decode_wall_times.append(time.time() - worker_start)
            decoded_batches.put(("done",))

    def finish_io_workers():
        for worker in io_workers:
            worker.join()
        for _ in range(decode_threads):
            raw_batches.put(("done",))

    io_workers = [threading.Thread(target=io_worker) for _ in range(io_threads)]
    decode_workers = [threading.Thread(target=decode_worker) for _ in range(decode_threads)]
    for worker in io_workers:
        worker.start()
    for worker in decode_workers:
        worker.start()
    finisher = threading.Thread(target=finish_io_workers)
    finisher.start()

    done_decoders = 0
    join_seconds = 0.0
    group_seconds = 0.0
    row_count = 0

    while done_decoders < decode_threads:
        item = decoded_batches.get()
        item_type = item[0]
        if item_type == "done":
            done_decoders += 1
            continue
        if item_type == "error":
            finisher.join()
            for worker in decode_workers:
                worker.join()
            raise item[1]

        _, _, batch_paths, fact, batch_profile = item

        if shared_dim is None:
            dimension_path = resolve_dimension_path(batch_paths[0], args.dimension_file)
            dim, dim_profile = timed_read_parquet_gpu(
                cudf,
                dimension_path,
                columns=dimension_columns,
                measure_materialized_bytes=measure_materialized_bytes,
            )
            detail_timers["dimension_read"] += dim_profile["elapsed"]
            detail_timers["dimension_rchar"] += dim_profile["rchar"]
            detail_timers["dimension_read_bytes"] += dim_profile["read_bytes"]
            detail_timers["dimension_syscr"] += dim_profile["syscr"]
        else:
            dim = shared_dim

        join_start = time.time()
        if args.infer_grid_from_row_order:
            joined = attach_group_from_row_order(cudf, fact, dim, args.group_column)
        else:
            joined = fact.merge(dim, on=args.join_key, how="inner")
        cuda_synchronize()
        join_elapsed = time.time() - join_start
        join_seconds += join_elapsed
        detail_timers["join"] += join_elapsed
        batch_profile["join"] = join_elapsed
        row_count += len(joined)

        group_start = time.time()
        sums, counts, row_counts = aggregate_joined(
            joined,
            args.group_column,
            payload_columns,
            args.assume_payload_all_valid,
            detail_timers,
            need_counts=need_counts,
            need_row_counts=need_row_counts,
        )
        cuda_synchronize()
        group_elapsed = time.time() - group_start
        group_seconds += group_elapsed
        detail_timers["aggregate_wall"] += group_elapsed
        batch_profile["groupby"] = group_elapsed
        append_frame(sum_frames, sums)
        append_frame(count_frames, counts)
        append_frame(row_count_frames, row_counts)
        batch_profiles.append(batch_profile)
        del fact
        del joined

    finisher.join()
    for worker in decode_workers:
        worker.join()

    detail_timers["file_io_wall"] = max(io_wall_times) if io_wall_times else 0.0
    detail_timers["decode_wall"] = max(decode_wall_times) if decode_wall_times else 0.0
    detail_timers["reader_wall"] = detail_timers["file_io_wall"]
    read_seconds = detail_timers["dimension_read"] + max(detail_timers["file_io_wall"], detail_timers["decode_wall"])
    return read_seconds, join_seconds, group_seconds, row_count


def combine_frames(cudf, frames, group_column, payload_columns, timers, is_row_count=False):
    if not frames:
        columns = [group_column, "row_count"] if is_row_count else [group_column] + payload_columns
        return cudf.DataFrame({column: [] for column in columns})
    timer_start = time.time()
    combined = cudf.concat(frames, ignore_index=True)
    cuda_synchronize()
    add_elapsed(timers, "merge_concat", timer_start)

    timer_start = time.time()
    if is_row_count:
        result = combined.groupby(group_column).sum().reset_index()
    else:
        result = combined.groupby(group_column)[payload_columns].sum().reset_index()
    cuda_synchronize()
    add_elapsed(timers, "merge_groupby", timer_start)
    return result


def main():
    args = parse_args()
    detail_timers = new_timers()
    rmm_pool_enabled, rmm_pool_size = configure_rmm_pool(args, detail_timers)
    try:
        import cudf
    except ImportError as exc:
        raise SystemExit("cudf is required for GPU-native Parquet scan: {}".format(exc))

    parquet_paths = sorted(glob.glob(os.path.join(args.base_dir, "UP-*", "time-levs-grid.parquet")))
    if not parquet_paths:
        raise SystemExit("no time-levs-grid.parquet files found")

    payload_columns = resolve_payload_columns(args, parquet_paths)
    if not payload_columns:
        raise SystemExit("no payload columns specified")

    read_seconds = 0.0
    join_seconds = 0.0
    group_seconds = 0.0
    merge_seconds = 0.0
    row_count = 0

    start = time.time()
    sum_frames = []
    count_frames = []
    row_count_frames = []
    batch_profiles = []

    if args.reader_threads <= 0:
        raise SystemExit("--reader-threads must be positive")
    if args.pipeline_queue_depth <= 0:
        raise SystemExit("--pipeline-queue-depth must be positive")
    if args.io_queue_depth <= 0:
        raise SystemExit("--io-queue-depth must be positive")
    if args.decode_threads <= 0:
        raise SystemExit("--decode-threads must be positive")

    if args.split_file_io and args.read_mode != "per-file":
        raise SystemExit("--split-file-io requires --read-mode per-file")

    if args.split_file_io:
        read_seconds, join_seconds, group_seconds, row_count = split_file_io_read_aggregate(
            cudf,
            args,
            parquet_paths,
            payload_columns,
            detail_timers,
            sum_frames,
            count_frames,
            row_count_frames,
            batch_profiles,
        )
    elif args.read_mode == "per-file":
        read_seconds, join_seconds, group_seconds, row_count = threaded_read_aggregate(
            cudf,
            args,
            parquet_paths,
            payload_columns,
            detail_timers,
            sum_frames,
            count_frames,
            row_count_frames,
            batch_profiles,
        )
    elif args.read_mode == "glob":
        dimension_path = resolve_dimension_path(parquet_paths[0], args.dimension_file)
        read_start = time.time()
        dimension_columns = [args.group_column] if args.infer_grid_from_row_order else [args.join_key, args.group_column]
        dim, dim_profile = timed_read_parquet_gpu(
            cudf,
            dimension_path,
            columns=dimension_columns,
            measure_materialized_bytes=args.print_stage_times or args.print_batch_times,
        )
        detail_timers["dimension_read"] += dim_profile["elapsed"]
        detail_timers["dimension_rchar"] += dim_profile["rchar"]
        detail_timers["dimension_read_bytes"] += dim_profile["read_bytes"]
        detail_timers["dimension_syscr"] += dim_profile["syscr"]

        fact_columns = payload_columns if args.infer_grid_from_row_order else [args.join_key] + payload_columns
        fact, fact_profile = timed_read_parquet_gpu(
            cudf,
            parquet_paths,
            columns=fact_columns,
            measure_materialized_bytes=args.print_stage_times or args.print_batch_times,
        )
        detail_timers["fact_read"] += fact_profile["elapsed"]
        detail_timers["fact_rchar"] += fact_profile["rchar"]
        detail_timers["fact_read_bytes"] += fact_profile["read_bytes"]
        detail_timers["fact_syscr"] += fact_profile["syscr"]
        detail_timers["materialized_bytes"] += fact_profile["materialized_bytes"]
        read_seconds += time.time() - read_start
        batch_profile = {
            "batch_id": 0,
            "files": len(parquet_paths),
            "read": fact_profile["elapsed"],
            "file_io": 0.0,
            "decode": fact_profile["elapsed"],
            "io_method": "cudf",
            "io_fallback_reason": "",
            "rows": fact_profile["rows"],
            "rchar": fact_profile["rchar"],
            "read_bytes": fact_profile["read_bytes"],
            "syscr": fact_profile["syscr"],
            "materialized_bytes": fact_profile["materialized_bytes"],
            "join": 0.0,
            "groupby": 0.0,
        }

        if args.infer_grid_from_row_order:
            join_start = time.time()
            joined = attach_group_from_row_order(cudf, fact, dim, args.group_column)
            cuda_synchronize()
            join_elapsed = time.time() - join_start
            join_seconds += join_elapsed
            detail_timers["join"] += join_elapsed
        else:
            join_start = time.time()
            joined = fact.merge(dim, on=args.join_key, how="inner")
            cuda_synchronize()
            join_elapsed = time.time() - join_start
            join_seconds += join_elapsed
            detail_timers["join"] += join_elapsed
        batch_profile["join"] = join_elapsed
        row_count = len(joined)

        group_start = time.time()
        sums, counts, row_counts = aggregate_joined(
            joined,
            args.group_column,
            payload_columns,
            args.assume_payload_all_valid,
            detail_timers,
            need_counts=args.print_results,
            need_row_counts=args.print_results,
        )
        cuda_synchronize()
        group_elapsed = time.time() - group_start
        group_seconds += group_elapsed
        detail_timers["aggregate_wall"] += group_elapsed
        batch_profile["groupby"] = group_elapsed
        append_frame(sum_frames, sums)
        append_frame(count_frames, counts)
        append_frame(row_count_frames, row_counts)
        batch_profiles.append(batch_profile)
        del fact
        del joined

    merge_start = time.time()
    total_sums = combine_frames(cudf, sum_frames, args.group_column, payload_columns, detail_timers)
    total_counts = None
    total_row_counts = None
    if args.print_results:
        total_counts = combine_frames(cudf, count_frames, args.group_column, payload_columns, detail_timers)
        total_row_counts = combine_frames(cudf, row_count_frames, args.group_column, payload_columns, detail_timers, True)
    cuda_synchronize()
    if args.print_results:
        timer_start = time.time()
        row_count = int(total_row_counts["row_count"].sum()) if len(total_row_counts) else 0
        cuda_synchronize()
        add_elapsed(detail_timers, "row_count_sum", timer_start)
    # Keep final aggregate columns alive until after the last GPU operation has completed.
    _ = (total_sums, total_counts)
    merge_seconds += time.time() - merge_start

    elapsed = time.time() - start
    print("\n[Row Count]")
    print(row_count)
    print("[number of input file]: {}".format(len(parquet_paths)))
    print("[read mode]: {}".format(args.read_mode))
    print("[scan/decode engine]: cudf/libcudf")
    print("[reader threads]: {}".format(max(1, min(args.reader_threads, len(parquet_paths)))))
    if rmm_pool_enabled:
        print("[rmm pool]: on size={} bytes".format(rmm_pool_size))
    if args.read_mode == "per-file":
        batch_size = resolve_file_batch_size(args, max(1, min(args.reader_threads, len(parquet_paths))), len(parquet_paths))
        print("[file batch size]: {}".format(batch_size))
        print("[pipeline queue depth]: {}".format(args.pipeline_queue_depth))
    if args.split_file_io:
        print("[split file io]: on")
        print("[file io method]: {}".format(args.file_io_method))
        if args.file_io_method == "io_uring":
            print("[file io sqpoll]: {}".format("on" if args.file_io_sqpoll else "off"))
            if args.file_io_sqpoll:
                print("[file io sqpoll idle ms]: {}".format(args.file_io_sqpoll_idle_ms))
        print("[io queue depth]: {}".format(args.io_queue_depth))
        print("[decode threads]: {}".format(args.decode_threads))
    if args.assume_payload_all_valid:
        print("[assume payload all valid]: on")
    if args.infer_grid_from_row_order:
        print("[grid inference]: row-order")
    print("[payload columns]: {}".format(",".join(payload_columns)))
    if args.print_stage_times:
        print(
            "[stage work] gpu_read={:.6f}s gpu_join={:.6f}s gpu_groupby={:.6f}s gpu_merge={:.6f}s".format(
                read_seconds, join_seconds, group_seconds, merge_seconds
            )
        )
        print(
            "[stage detail] dimension_read={:.6f}s fact_read={:.6f}s join={:.6f}s "
            "groupby_size={:.6f}s groupby_sum={:.6f}s groupby_count={:.6f}s "
            "count_from_rows={:.6f}s merge_concat={:.6f}s merge_groupby={:.6f}s "
            "row_count_sum={:.6f}s reader_wall={:.6f}s aggregate_wall={:.6f}s "
            "file_io_read={:.6f}s file_io_wall={:.6f}s decode_materialize={:.6f}s "
            "decode_wall={:.6f}s io_uring_fallbacks={} rmm_setup={:.6f}s".format(
                detail_timers["dimension_read"],
                detail_timers["fact_read"],
                detail_timers["join"],
                detail_timers["groupby_size"],
                detail_timers["groupby_sum"],
                detail_timers["groupby_count"],
                detail_timers["count_from_rows"],
                detail_timers["merge_concat"],
                detail_timers["merge_groupby"],
                detail_timers["row_count_sum"],
                detail_timers["reader_wall"],
                detail_timers["aggregate_wall"],
                detail_timers["file_io_read"],
                detail_timers["file_io_wall"],
                detail_timers["decode_materialize"],
                detail_timers["decode_wall"],
                detail_timers["io_uring_fallbacks"],
                detail_timers["rmm_setup"],
            )
        )
        if detail_timers["fact_read"] > 0:
            read_mib = float(detail_timers["fact_read_bytes"]) / (1024.0 * 1024.0)
            rchar_mib = float(detail_timers["fact_rchar"]) / (1024.0 * 1024.0)
            materialized_mib = float(detail_timers["materialized_bytes"]) / (1024.0 * 1024.0)
            read_rate_seconds = detail_timers["file_io_read"] or detail_timers["fact_read"]
            print(
                "[read io] fact_read_bytes={} ({:.2f} MiB) fact_rchar={} ({:.2f} MiB) "
                "fact_syscr={} materialized_bytes={} ({:.2f} MiB) storage_read_rate={:.2f} MiB/s".format(
                    detail_timers["fact_read_bytes"],
                    read_mib,
                    detail_timers["fact_rchar"],
                    rchar_mib,
                    detail_timers["fact_syscr"],
                    detail_timers["materialized_bytes"],
                    materialized_mib,
                    read_mib / read_rate_seconds,
                )
            )
        if detail_timers["dimension_read"] > 0:
            print(
                "[dimension io] read_bytes={} rchar={} syscr={}".format(
                    detail_timers["dimension_read_bytes"],
                    detail_timers["dimension_rchar"],
                    detail_timers["dimension_syscr"],
                )
            )
    if args.print_batch_times:
        print("\n[Batch Times]")
        for profile in sorted(batch_profiles, key=lambda item: item["batch_id"]):
            print(
                "batch={batch_id} files={files} rows={rows} read={read:.6f}s "
                "file_io={file_io:.6f}s decode={decode:.6f}s io_method={io_method} "
                "join={join:.6f}s "
                "groupby={groupby:.6f}s read_bytes={read_bytes} rchar={rchar} "
                "syscr={syscr} materialized_bytes={materialized_bytes}".format(**profile)
            )
            if profile.get("io_fallback_reason"):
                print("  fallback_reason={}".format(profile["io_fallback_reason"]))
    if args.print_results:
        print("\n[Results]")
        merged = total_sums.merge(total_counts, on=args.group_column, suffixes=("_sum", "_count"))
        merged = merged.merge(total_row_counts, on=args.group_column)
        for row in merged.sort_values(args.group_column).to_pandas().itertuples(index=False):
            print(row)
    print("[query time]: {:.6f}s".format(elapsed))


if __name__ == "__main__":
    main()
