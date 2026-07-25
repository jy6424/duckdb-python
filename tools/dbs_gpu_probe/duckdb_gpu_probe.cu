#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace {

__global__ void DuckDBGpuProbeI64Kernel(const int64_t *keys, const uint8_t *validity, uint64_t count,
                                        int64_t min_value, int64_t max_value, const uint8_t *build_bitmap,
                                        uint64_t build_size, uint32_t *probe_sel_out, uint32_t *build_sel_out,
                                        unsigned long long *out_count) {
	const auto row = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (row >= count) {
		return;
	}
	if (validity && validity[row] == 0) {
		return;
	}

	const auto key = keys[row];
	if (key < min_value || key > max_value) {
		return;
	}

	const auto build_idx = static_cast<uint64_t>(key - min_value);
	if (build_idx >= build_size || build_bitmap[build_idx] == 0) {
		return;
	}

	const auto out_idx = atomicAdd(out_count, 1ULL);
	probe_sel_out[out_idx] = static_cast<uint32_t>(row);
	build_sel_out[out_idx] = static_cast<uint32_t>(build_idx);
}

__global__ void DuckDBGpuProbeU16Kernel(const uint16_t *keys, const uint8_t *validity, uint64_t count,
                                        uint16_t min_value, uint16_t max_value, const uint8_t *build_bitmap,
                                        uint64_t build_size, uint32_t *probe_sel_out, uint32_t *build_sel_out,
                                        unsigned long long *out_count) {
	const auto row = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (row >= count) {
		return;
	}
	if (validity && validity[row] == 0) {
		return;
	}

	const auto key = keys[row];
	if (key < min_value || key > max_value) {
		return;
	}

	const auto build_idx = static_cast<uint64_t>(key - min_value);
	if (build_idx >= build_size || build_bitmap[build_idx] == 0) {
		return;
	}

	const auto out_idx = atomicAdd(out_count, 1ULL);
	probe_sel_out[out_idx] = static_cast<uint32_t>(row);
	build_sel_out[out_idx] = static_cast<uint32_t>(build_idx);
}

__global__ void DuckDBGpuGroupByCountKernel(const uint64_t *addresses, const uint8_t *validity, uint64_t count,
                                            uint64_t *unique_addresses_out, uint64_t *counts_out,
                                            unsigned long long *unique_count_out) {
	const auto row = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (row >= count || validity[row] == 0) {
		return;
	}

	const auto address = addresses[row];
	for (uint64_t i = 0; i < row; i++) {
		if (validity[i] != 0 && addresses[i] == address) {
			return;
		}
	}

	uint64_t local_count = 0;
	for (uint64_t i = row; i < count; i++) {
		if (validity[i] != 0 && addresses[i] == address) {
			local_count++;
		}
	}

	const auto out_idx = atomicAdd(unique_count_out, 1ULL);
	unique_addresses_out[out_idx] = address;
	counts_out[out_idx] = local_count;
}

int CheckCuda(cudaError_t status, const char *step) {
	if (status == cudaSuccess) {
		return 0;
	}
	std::fprintf(stderr, "[duckdb gpu offload] CUDA %s failed: %s\n", step, cudaGetErrorString(status));
	return 1;
}

struct DeviceBuffer {
	void *ptr = nullptr;
	size_t capacity = 0;

	~DeviceBuffer() {
		if (ptr) {
			cudaFree(ptr);
		}
	}

	int Ensure(size_t bytes, const char *step) {
		if (bytes <= capacity) {
			return 0;
		}
		if (ptr) {
			cudaFree(ptr);
			ptr = nullptr;
			capacity = 0;
		}
		if (CheckCuda(cudaMalloc(&ptr, bytes), step)) {
			return 1;
		}
		capacity = bytes;
		return 0;
	}

	template <class T>
	T *As() {
		return reinterpret_cast<T *>(ptr);
	}
};

struct ProbeI64Buffers {
	DeviceBuffer keys;
	DeviceBuffer validity;
	DeviceBuffer build_bitmap;
	DeviceBuffer probe_sel;
	DeviceBuffer build_sel;
	DeviceBuffer count;
};

struct ProbeU16Buffers {
	DeviceBuffer keys;
	DeviceBuffer validity;
	DeviceBuffer build_bitmap;
	DeviceBuffer probe_sel;
	DeviceBuffer build_sel;
	DeviceBuffer count;
};

struct GroupByCountBuffers {
	DeviceBuffer addresses;
	DeviceBuffer validity;
	DeviceBuffer unique_addresses;
	DeviceBuffer counts;
	DeviceBuffer unique_count;
};

} // namespace

extern "C" int duckdb_gpu_probe_i64(const int64_t *keys, const uint8_t *validity, uint64_t count, int64_t min_value,
                                    int64_t max_value, const uint8_t *build_bitmap, uint64_t build_size,
                                    uint32_t *probe_sel_out, uint32_t *build_sel_out, uint64_t *out_count) {
	if (!keys || !validity || !build_bitmap || !probe_sel_out || !build_sel_out || !out_count) {
		return 1;
	}
	if (count == 0) {
		*out_count = 0;
		return 0;
	}

	unsigned long long result_count = 0;
	thread_local ProbeI64Buffers buffers;

	const auto keys_bytes = count * sizeof(int64_t);
	const auto validity_bytes = count * sizeof(uint8_t);
	const auto bitmap_bytes = build_size * sizeof(uint8_t);
	const auto output_bytes = count * sizeof(uint32_t);

	int error = 0;
	error |= buffers.keys.Ensure(keys_bytes, "resize keys");
	error |= buffers.validity.Ensure(validity_bytes, "resize validity");
	error |= buffers.build_bitmap.Ensure(bitmap_bytes, "resize build bitmap");
	error |= buffers.probe_sel.Ensure(output_bytes, "resize probe selection");
	error |= buffers.build_sel.Ensure(output_bytes, "resize build selection");
	error |= buffers.count.Ensure(sizeof(unsigned long long), "resize count");
	if (error) {
		return 1;
	}

	auto d_keys = buffers.keys.As<int64_t>();
	auto d_validity = buffers.validity.As<uint8_t>();
	auto d_build_bitmap = buffers.build_bitmap.As<uint8_t>();
	auto d_probe_sel = buffers.probe_sel.As<uint32_t>();
	auto d_build_sel = buffers.build_sel.As<uint32_t>();
	auto d_count = buffers.count.As<unsigned long long>();

	error |= CheckCuda(cudaMemcpy(d_keys, keys, keys_bytes, cudaMemcpyHostToDevice), "copy keys to device");
	error |=
	    CheckCuda(cudaMemcpy(d_validity, validity, validity_bytes, cudaMemcpyHostToDevice), "copy validity to device");
	error |= CheckCuda(cudaMemcpy(d_build_bitmap, build_bitmap, bitmap_bytes, cudaMemcpyHostToDevice),
	                   "copy build bitmap to device");
	error |= CheckCuda(cudaMemset(d_count, 0, sizeof(unsigned long long)), "clear count");
	if (error) {
		return 1;
	}

	{
		constexpr int THREADS_PER_BLOCK = 256;
		const auto blocks = static_cast<unsigned int>((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
		DuckDBGpuProbeI64Kernel<<<blocks, THREADS_PER_BLOCK>>>(d_keys, d_validity, count, min_value, max_value,
		                                                       d_build_bitmap, build_size, d_probe_sel, d_build_sel,
		                                                       d_count);
		error |= CheckCuda(cudaGetLastError(), "launch probe kernel");
	}
	if (error) {
		return 1;
	}

	error |= CheckCuda(cudaMemcpy(&result_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost),
	                   "copy count to host");
	if (error || result_count > count) {
		return 1;
	}

	error |= CheckCuda(cudaMemcpy(probe_sel_out, d_probe_sel, result_count * sizeof(uint32_t), cudaMemcpyDeviceToHost),
	                   "copy probe selection to host");
	error |= CheckCuda(cudaMemcpy(build_sel_out, d_build_sel, result_count * sizeof(uint32_t), cudaMemcpyDeviceToHost),
	                   "copy build selection to host");
	*out_count = static_cast<uint64_t>(result_count);
	return error ? 1 : 0;
}

extern "C" int duckdb_gpu_groupby_count(const uint64_t *addresses, const uint8_t *validity, uint64_t count,
                                        uint64_t *unique_addresses_out, uint64_t *counts_out,
                                        uint64_t *unique_count_out) {
	if (!addresses || !validity || !unique_addresses_out || !counts_out || !unique_count_out) {
		return 1;
	}
	if (count == 0) {
		*unique_count_out = 0;
		return 0;
	}

	unsigned long long result_count = 0;
	thread_local GroupByCountBuffers buffers;

	const auto addresses_bytes = count * sizeof(uint64_t);
	const auto validity_bytes = count * sizeof(uint8_t);
	const auto output_bytes = count * sizeof(uint64_t);

	int error = 0;
	error |= buffers.addresses.Ensure(addresses_bytes, "resize groupby count addresses");
	error |= buffers.validity.Ensure(validity_bytes, "resize groupby count validity");
	error |= buffers.unique_addresses.Ensure(output_bytes, "resize groupby count unique addresses");
	error |= buffers.counts.Ensure(output_bytes, "resize groupby count output counts");
	error |= buffers.unique_count.Ensure(sizeof(unsigned long long), "resize groupby count unique count");
	if (error) {
		return 1;
	}

	auto d_addresses = buffers.addresses.As<uint64_t>();
	auto d_validity = buffers.validity.As<uint8_t>();
	auto d_unique_addresses = buffers.unique_addresses.As<uint64_t>();
	auto d_counts = buffers.counts.As<uint64_t>();
	auto d_unique_count = buffers.unique_count.As<unsigned long long>();

	error |= CheckCuda(cudaMemcpy(d_addresses, addresses, addresses_bytes, cudaMemcpyHostToDevice),
	                   "copy groupby count addresses to device");
	error |= CheckCuda(cudaMemcpy(d_validity, validity, validity_bytes, cudaMemcpyHostToDevice),
	                   "copy groupby count validity to device");
	error |= CheckCuda(cudaMemset(d_unique_count, 0, sizeof(unsigned long long)), "clear groupby count unique count");
	if (error) {
		return 1;
	}

	{
		constexpr int THREADS_PER_BLOCK = 256;
		const auto blocks = static_cast<unsigned int>((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
		DuckDBGpuGroupByCountKernel<<<blocks, THREADS_PER_BLOCK>>>(d_addresses, d_validity, count, d_unique_addresses,
		                                                           d_counts, d_unique_count);
		error |= CheckCuda(cudaGetLastError(), "launch groupby count kernel");
	}
	if (error) {
		return 1;
	}

	error |= CheckCuda(cudaMemcpy(&result_count, d_unique_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost),
	                   "copy groupby count unique count to host");
	if (error || result_count > count) {
		return 1;
	}

	error |= CheckCuda(cudaMemcpy(unique_addresses_out, d_unique_addresses, result_count * sizeof(uint64_t),
	                              cudaMemcpyDeviceToHost),
	                   "copy groupby count unique addresses to host");
	error |= CheckCuda(cudaMemcpy(counts_out, d_counts, result_count * sizeof(uint64_t), cudaMemcpyDeviceToHost),
	                   "copy groupby count counts to host");
	*unique_count_out = static_cast<uint64_t>(result_count);
	return error ? 1 : 0;
}

extern "C" int duckdb_gpu_probe_u16(const uint16_t *keys, const uint8_t *validity, uint64_t count, uint16_t min_value,
                                    uint16_t max_value, const uint8_t *build_bitmap, uint64_t build_size,
                                    uint32_t *probe_sel_out, uint32_t *build_sel_out, uint64_t *out_count) {
	if (!keys || !validity || !build_bitmap || !probe_sel_out || !build_sel_out || !out_count) {
		return 1;
	}
	if (count == 0) {
		*out_count = 0;
		return 0;
	}

	unsigned long long result_count = 0;
	thread_local ProbeU16Buffers buffers;

	const auto keys_bytes = count * sizeof(uint16_t);
	const auto validity_bytes = count * sizeof(uint8_t);
	const auto bitmap_bytes = build_size * sizeof(uint8_t);
	const auto output_bytes = count * sizeof(uint32_t);

	int error = 0;
	error |= buffers.keys.Ensure(keys_bytes, "resize u16 keys");
	error |= buffers.validity.Ensure(validity_bytes, "resize u16 validity");
	error |= buffers.build_bitmap.Ensure(bitmap_bytes, "resize u16 build bitmap");
	error |= buffers.probe_sel.Ensure(output_bytes, "resize u16 probe selection");
	error |= buffers.build_sel.Ensure(output_bytes, "resize u16 build selection");
	error |= buffers.count.Ensure(sizeof(unsigned long long), "resize u16 count");
	if (error) {
		return 1;
	}

	auto d_keys = buffers.keys.As<uint16_t>();
	auto d_validity = buffers.validity.As<uint8_t>();
	auto d_build_bitmap = buffers.build_bitmap.As<uint8_t>();
	auto d_probe_sel = buffers.probe_sel.As<uint32_t>();
	auto d_build_sel = buffers.build_sel.As<uint32_t>();
	auto d_count = buffers.count.As<unsigned long long>();

	error |= CheckCuda(cudaMemcpy(d_keys, keys, keys_bytes, cudaMemcpyHostToDevice), "copy u16 keys to device");
	error |= CheckCuda(cudaMemcpy(d_validity, validity, validity_bytes, cudaMemcpyHostToDevice),
	                   "copy u16 validity to device");
	error |= CheckCuda(cudaMemcpy(d_build_bitmap, build_bitmap, bitmap_bytes, cudaMemcpyHostToDevice),
	                   "copy u16 build bitmap to device");
	error |= CheckCuda(cudaMemset(d_count, 0, sizeof(unsigned long long)), "clear u16 count");
	if (error) {
		return 1;
	}

	{
		constexpr int THREADS_PER_BLOCK = 256;
		const auto blocks = static_cast<unsigned int>((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
		DuckDBGpuProbeU16Kernel<<<blocks, THREADS_PER_BLOCK>>>(d_keys, d_validity, count, min_value, max_value,
		                                                       d_build_bitmap, build_size, d_probe_sel, d_build_sel,
		                                                       d_count);
		error |= CheckCuda(cudaGetLastError(), "launch u16 probe kernel");
	}
	if (error) {
		return 1;
	}

	error |= CheckCuda(cudaMemcpy(&result_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost),
	                   "copy u16 count to host");
	if (error || result_count > count) {
		return 1;
	}

	error |= CheckCuda(cudaMemcpy(probe_sel_out, d_probe_sel, result_count * sizeof(uint32_t), cudaMemcpyDeviceToHost),
	                   "copy u16 probe selection to host");
	error |= CheckCuda(cudaMemcpy(build_sel_out, d_build_sel, result_count * sizeof(uint32_t), cudaMemcpyDeviceToHost),
	                   "copy u16 build selection to host");
	*out_count = static_cast<uint64_t>(result_count);
	return error ? 1 : 0;
}
