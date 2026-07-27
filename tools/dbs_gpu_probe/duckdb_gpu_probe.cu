#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

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

__global__ void DuckDBGpuFusedLatAggKernel(const int64_t *grids, const double *values, const uint8_t *value_validity,
                                           uint64_t count, int64_t grid_min, int64_t grid_max,
                                           const int32_t *grid_to_group, uint64_t build_size, double *sum_out,
                                           unsigned long long *count_out, unsigned long long *row_count_out) {
	const auto row = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (row >= count) {
		return;
	}

	const auto grid = grids[row];
	if (grid < grid_min || grid > grid_max) {
		return;
	}

	const auto build_idx = static_cast<uint64_t>(grid - grid_min);
	if (build_idx >= build_size) {
		return;
	}

	const auto group = grid_to_group[build_idx];
	if (group < 0) {
		return;
	}

	atomicAdd(&row_count_out[group], 1ULL);
	if (!value_validity || value_validity[row] != 0) {
		atomicAdd(&sum_out[group], values[row]);
		atomicAdd(&count_out[group], 1ULL);
	}
}

int CheckCuda(cudaError_t status, const char *step) {
	if (status == cudaSuccess) {
		return 0;
	}
	std::fprintf(stderr, "[duckdb gpu offload] CUDA %s failed: %s\n", step, cudaGetErrorString(status));
	return 1;
}

struct MappedHostBuffer {
	void *host_ptr = nullptr;
	void *device_ptr = nullptr;
	size_t capacity = 0;

	~MappedHostBuffer() {
		if (host_ptr) {
			cudaFreeHost(host_ptr);
		}
	}

	int Ensure(size_t bytes, const char *step) {
		if (bytes <= capacity) {
			return 0;
		}
		if (host_ptr) {
			cudaFreeHost(host_ptr);
			host_ptr = nullptr;
			device_ptr = nullptr;
			capacity = 0;
		}
		if (CheckCuda(cudaHostAlloc(&host_ptr, bytes, cudaHostAllocMapped), step)) {
			return 1;
		}
		if (CheckCuda(cudaHostGetDevicePointer(&device_ptr, host_ptr, 0), step)) {
			cudaFreeHost(host_ptr);
			host_ptr = nullptr;
			device_ptr = nullptr;
			return 1;
		}
		capacity = bytes;
		return 0;
	}

	template <class T>
	T *HostAs() {
		return reinterpret_cast<T *>(host_ptr);
	}

	template <class T>
	T *DeviceAs() {
		return reinterpret_cast<T *>(device_ptr);
	}
};

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
	MappedHostBuffer keys;
	MappedHostBuffer validity;
	MappedHostBuffer build_bitmap;
	MappedHostBuffer probe_sel;
	MappedHostBuffer build_sel;
	MappedHostBuffer count;
};

struct ProbeU16Buffers {
	MappedHostBuffer keys;
	MappedHostBuffer validity;
	MappedHostBuffer build_bitmap;
	MappedHostBuffer probe_sel;
	MappedHostBuffer build_sel;
	MappedHostBuffer count;
};

struct GroupByCountBuffers {
	MappedHostBuffer addresses;
	MappedHostBuffer validity;
	MappedHostBuffer unique_addresses;
	MappedHostBuffer counts;
	MappedHostBuffer unique_count;
};

struct FusedLatAggBuffers {
	MappedHostBuffer grids;
	MappedHostBuffer values;
	MappedHostBuffer value_validity;
	MappedHostBuffer grid_to_group;
	DeviceBuffer sums;
	DeviceBuffer counts;
	DeviceBuffer row_counts;
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

	auto h_keys = buffers.keys.HostAs<int64_t>();
	auto h_validity = buffers.validity.HostAs<uint8_t>();
	auto h_build_bitmap = buffers.build_bitmap.HostAs<uint8_t>();
	auto h_probe_sel = buffers.probe_sel.HostAs<uint32_t>();
	auto h_build_sel = buffers.build_sel.HostAs<uint32_t>();
	auto h_count = buffers.count.HostAs<unsigned long long>();
	auto d_keys = buffers.keys.DeviceAs<int64_t>();
	auto d_validity = buffers.validity.DeviceAs<uint8_t>();
	auto d_build_bitmap = buffers.build_bitmap.DeviceAs<uint8_t>();
	auto d_probe_sel = buffers.probe_sel.DeviceAs<uint32_t>();
	auto d_build_sel = buffers.build_sel.DeviceAs<uint32_t>();
	auto d_count = buffers.count.DeviceAs<unsigned long long>();

	std::memcpy(h_keys, keys, keys_bytes);
	std::memcpy(h_validity, validity, validity_bytes);
	std::memcpy(h_build_bitmap, build_bitmap, bitmap_bytes);
	*h_count = 0;

	{
		constexpr int THREADS_PER_BLOCK = 256;
		const auto blocks = static_cast<unsigned int>((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
		DuckDBGpuProbeI64Kernel<<<blocks, THREADS_PER_BLOCK>>>(d_keys, d_validity, count, min_value, max_value,
		                                                       d_build_bitmap, build_size, d_probe_sel, d_build_sel,
		                                                       d_count);
		error |= CheckCuda(cudaGetLastError(), "launch probe kernel");
		error |= CheckCuda(cudaDeviceSynchronize(), "synchronize mapped probe kernel");
	}
	if (error) {
		return 1;
	}

	result_count = *h_count;
	if (error || result_count > count) {
		return 1;
	}

	std::memcpy(probe_sel_out, h_probe_sel, result_count * sizeof(uint32_t));
	std::memcpy(build_sel_out, h_build_sel, result_count * sizeof(uint32_t));
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

	auto h_addresses = buffers.addresses.HostAs<uint64_t>();
	auto h_validity = buffers.validity.HostAs<uint8_t>();
	auto h_unique_addresses = buffers.unique_addresses.HostAs<uint64_t>();
	auto h_counts = buffers.counts.HostAs<uint64_t>();
	auto h_unique_count = buffers.unique_count.HostAs<unsigned long long>();
	auto d_addresses = buffers.addresses.DeviceAs<uint64_t>();
	auto d_validity = buffers.validity.DeviceAs<uint8_t>();
	auto d_unique_addresses = buffers.unique_addresses.DeviceAs<uint64_t>();
	auto d_counts = buffers.counts.DeviceAs<uint64_t>();
	auto d_unique_count = buffers.unique_count.DeviceAs<unsigned long long>();

	std::memcpy(h_addresses, addresses, addresses_bytes);
	std::memcpy(h_validity, validity, validity_bytes);
	*h_unique_count = 0;

	{
		constexpr int THREADS_PER_BLOCK = 256;
		const auto blocks = static_cast<unsigned int>((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
		DuckDBGpuGroupByCountKernel<<<blocks, THREADS_PER_BLOCK>>>(d_addresses, d_validity, count, d_unique_addresses,
		                                                           d_counts, d_unique_count);
		error |= CheckCuda(cudaGetLastError(), "launch groupby count kernel");
		error |= CheckCuda(cudaDeviceSynchronize(), "synchronize mapped groupby count kernel");
	}
	if (error) {
		return 1;
	}

	result_count = *h_unique_count;
	if (error || result_count > count) {
		return 1;
	}

	std::memcpy(unique_addresses_out, h_unique_addresses, result_count * sizeof(uint64_t));
	std::memcpy(counts_out, h_counts, result_count * sizeof(uint64_t));
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

	auto h_keys = buffers.keys.HostAs<uint16_t>();
	auto h_validity = buffers.validity.HostAs<uint8_t>();
	auto h_build_bitmap = buffers.build_bitmap.HostAs<uint8_t>();
	auto h_probe_sel = buffers.probe_sel.HostAs<uint32_t>();
	auto h_build_sel = buffers.build_sel.HostAs<uint32_t>();
	auto h_count = buffers.count.HostAs<unsigned long long>();
	auto d_keys = buffers.keys.DeviceAs<uint16_t>();
	auto d_validity = buffers.validity.DeviceAs<uint8_t>();
	auto d_build_bitmap = buffers.build_bitmap.DeviceAs<uint8_t>();
	auto d_probe_sel = buffers.probe_sel.DeviceAs<uint32_t>();
	auto d_build_sel = buffers.build_sel.DeviceAs<uint32_t>();
	auto d_count = buffers.count.DeviceAs<unsigned long long>();

	std::memcpy(h_keys, keys, keys_bytes);
	std::memcpy(h_validity, validity, validity_bytes);
	std::memcpy(h_build_bitmap, build_bitmap, bitmap_bytes);
	*h_count = 0;

	{
		constexpr int THREADS_PER_BLOCK = 256;
		const auto blocks = static_cast<unsigned int>((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
		DuckDBGpuProbeU16Kernel<<<blocks, THREADS_PER_BLOCK>>>(d_keys, d_validity, count, min_value, max_value,
		                                                       d_build_bitmap, build_size, d_probe_sel, d_build_sel,
		                                                       d_count);
		error |= CheckCuda(cudaGetLastError(), "launch u16 probe kernel");
		error |= CheckCuda(cudaDeviceSynchronize(), "synchronize mapped u16 probe kernel");
	}
	if (error) {
		return 1;
	}

	result_count = *h_count;
	if (error || result_count > count) {
		return 1;
	}

	std::memcpy(probe_sel_out, h_probe_sel, result_count * sizeof(uint32_t));
	std::memcpy(build_sel_out, h_build_sel, result_count * sizeof(uint32_t));
	*out_count = static_cast<uint64_t>(result_count);
	return error ? 1 : 0;
}

extern "C" int duckdb_gpu_fused_lat_agg_i64_double(const int64_t *grids, const double *values,
                                                   const uint8_t *value_validity, uint64_t count, int64_t grid_min,
                                                   int64_t grid_max, const int32_t *grid_to_group,
                                                   uint64_t build_size, uint64_t group_count, double *sum_out,
                                                   uint64_t *count_out, uint64_t *row_count_out) {
	if (!grids || !values || !grid_to_group || !sum_out || !count_out || !row_count_out) {
		return 1;
	}
	if (group_count == 0) {
		return 0;
	}
	if (count == 0) {
		std::memset(sum_out, 0, group_count * sizeof(double));
		std::memset(count_out, 0, group_count * sizeof(uint64_t));
		std::memset(row_count_out, 0, group_count * sizeof(uint64_t));
		return 0;
	}
	if (grid_max < grid_min || build_size == 0) {
		return 1;
	}

	thread_local FusedLatAggBuffers buffers;
	const auto grid_bytes = count * sizeof(int64_t);
	const auto value_bytes = count * sizeof(double);
	const auto validity_bytes = count * sizeof(uint8_t);
	const auto build_bytes = build_size * sizeof(int32_t);
	const auto sum_bytes = group_count * sizeof(double);
	const auto count_bytes = group_count * sizeof(unsigned long long);

	int error = 0;
	error |= buffers.grids.Ensure(grid_bytes, "resize fused grids");
	error |= buffers.values.Ensure(value_bytes, "resize fused values");
	error |= buffers.value_validity.Ensure(validity_bytes, "resize fused value validity");
	error |= buffers.grid_to_group.Ensure(build_bytes, "resize fused grid to group");
	error |= buffers.sums.Ensure(sum_bytes, "resize fused sums");
	error |= buffers.counts.Ensure(count_bytes, "resize fused counts");
	error |= buffers.row_counts.Ensure(count_bytes, "resize fused row counts");
	if (error) {
		return 1;
	}

	auto h_grids = buffers.grids.HostAs<int64_t>();
	auto h_values = buffers.values.HostAs<double>();
	auto h_value_validity = buffers.value_validity.HostAs<uint8_t>();
	auto h_grid_to_group = buffers.grid_to_group.HostAs<int32_t>();
	auto d_grids = buffers.grids.DeviceAs<int64_t>();
	auto d_values = buffers.values.DeviceAs<double>();
	auto d_value_validity = buffers.value_validity.DeviceAs<uint8_t>();
	auto d_grid_to_group = buffers.grid_to_group.DeviceAs<int32_t>();
	auto d_sums = buffers.sums.As<double>();
	auto d_counts = buffers.counts.As<unsigned long long>();
	auto d_row_counts = buffers.row_counts.As<unsigned long long>();

	std::memcpy(h_grids, grids, grid_bytes);
	std::memcpy(h_values, values, value_bytes);
	if (value_validity) {
		std::memcpy(h_value_validity, value_validity, validity_bytes);
	} else {
		std::memset(h_value_validity, 1, validity_bytes);
	}
	std::memcpy(h_grid_to_group, grid_to_group, build_bytes);

	error |= CheckCuda(cudaMemset(d_sums, 0, sum_bytes), "clear fused sums");
	error |= CheckCuda(cudaMemset(d_counts, 0, count_bytes), "clear fused counts");
	error |= CheckCuda(cudaMemset(d_row_counts, 0, count_bytes), "clear fused row counts");
	if (error) {
		return 1;
	}

	{
		constexpr int THREADS_PER_BLOCK = 256;
		const auto blocks = static_cast<unsigned int>((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
		DuckDBGpuFusedLatAggKernel<<<blocks, THREADS_PER_BLOCK>>>(d_grids, d_values, d_value_validity, count, grid_min,
		                                                          grid_max, d_grid_to_group, build_size, d_sums,
		                                                          d_counts, d_row_counts);
		error |= CheckCuda(cudaGetLastError(), "launch fused lat aggregate kernel");
		error |= CheckCuda(cudaDeviceSynchronize(), "synchronize fused lat aggregate kernel");
	}
	if (error) {
		return 1;
	}

	error |= CheckCuda(cudaMemcpy(sum_out, d_sums, sum_bytes, cudaMemcpyDeviceToHost), "copy fused sums to host");
	error |= CheckCuda(cudaMemcpy(count_out, d_counts, count_bytes, cudaMemcpyDeviceToHost),
	                   "copy fused counts to host");
	error |= CheckCuda(cudaMemcpy(row_count_out, d_row_counts, count_bytes, cudaMemcpyDeviceToHost),
	                   "copy fused row counts to host");
	return error ? 1 : 0;
}
