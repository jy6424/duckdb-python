#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

bool EnvFlag(const char *name) {
	const auto value = std::getenv(name);
	if (!value || value[0] == '\0') {
		return false;
	}
	return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 && std::strcmp(value, "FALSE") != 0 &&
	       std::strcmp(value, "off") != 0 && std::strcmp(value, "OFF") != 0;
}

bool AssumePayloadAllValid() {
	return EnvFlag("DUCKDB_GPU_ASSUME_PAYLOAD_ALL_VALID");
}

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

__global__ void DuckDBGpuGroupBySumDoubleKernel(const uint64_t *addresses, const double *values,
                                                const uint8_t *validity, uint64_t count,
                                                uint64_t *unique_addresses_out, double *sums_out,
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

	double local_sum = 0;
	for (uint64_t i = row; i < count; i++) {
		if (validity[i] != 0 && addresses[i] == address) {
			local_sum += values[i];
		}
	}

	const auto out_idx = atomicAdd(unique_count_out, 1ULL);
	unique_addresses_out[out_idx] = address;
	sums_out[out_idx] = local_sum;
}

__global__ void DuckDBGpuGroupByStatsDoubleKernel(const uint64_t *addresses, const double *values,
                                                  const uint8_t *validity, uint64_t count,
                                                  uint64_t *unique_addresses_out, double *sums_out,
                                                  unsigned long long *counts_out, double *mins_out, double *maxs_out,
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

	double local_sum = 0;
	double local_min = values[row];
	double local_max = values[row];
	unsigned long long local_count = 0;
	for (uint64_t i = row; i < count; i++) {
		if (validity[i] != 0 && addresses[i] == address) {
			const auto value = values[i];
			local_sum += value;
			local_count++;
			if (value < local_min) {
				local_min = value;
			}
			if (value > local_max) {
				local_max = value;
			}
		}
	}

	const auto out_idx = atomicAdd(unique_count_out, 1ULL);
	unique_addresses_out[out_idx] = address;
	sums_out[out_idx] = local_sum;
	counts_out[out_idx] = local_count;
	mins_out[out_idx] = local_min;
	maxs_out[out_idx] = local_max;
}

__device__ double AtomicAddDouble(double *address, double value);
__device__ double AtomicMinDouble(double *address, double value);
__device__ double AtomicMaxDouble(double *address, double value);

__global__ void DuckDBGpuInitDictStatsDoubleKernel(uint64_t group_count, double *sums_out,
                                                   unsigned long long *counts_out,
                                                   unsigned long long *row_counts_out, double *mins_out,
                                                   double *maxs_out) {
	const auto group = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (group >= group_count) {
		return;
	}
	sums_out[group] = 0;
	counts_out[group] = 0;
	row_counts_out[group] = 0;
	const auto positive_infinity = __longlong_as_double(0x7ff0000000000000ULL);
	mins_out[group] = positive_infinity;
	maxs_out[group] = -positive_infinity;
}

__global__ void DuckDBGpuGroupByDictStatsDoubleKernel(const uint32_t *group_ids, const double *values,
                                                      const uint8_t *validity, uint64_t count, uint64_t group_count,
                                                      double *sums_out, unsigned long long *counts_out,
                                                      unsigned long long *row_counts_out, double *mins_out,
                                                      double *maxs_out) {
	const auto row = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (row >= count) {
		return;
	}

	const auto group = static_cast<uint64_t>(group_ids[row]);
	if (group >= group_count) {
		return;
	}

	atomicAdd(&row_counts_out[group], 1ULL);
	if (validity && validity[row] == 0) {
		return;
	}

	const auto value = values[row];
	AtomicAddDouble(&sums_out[group], value);
	atomicAdd(&counts_out[group], 1ULL);
	AtomicMinDouble(&mins_out[group], value);
	AtomicMaxDouble(&maxs_out[group], value);
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
		AtomicAddDouble(&sum_out[group], values[row]);
		atomicAdd(&count_out[group], 1ULL);
	}
}

__device__ void DuckDBGpuFusedLatAggMultiRow(
    const int64_t *grids, const double *values, const uint8_t *value_validity, uint64_t column_count,
    uint64_t value_stride, uint64_t count, int64_t grid_min, int64_t grid_max, const int32_t *grid_to_group,
    uint64_t build_size, uint64_t group_count, double *sum_out, unsigned long long *count_out,
    unsigned long long *row_count_out, uint64_t row) {
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

	const auto group_i32 = grid_to_group[build_idx];
	if (group_i32 < 0) {
		return;
	}

	const auto group = static_cast<uint64_t>(group_i32);
	atomicAdd(&row_count_out[group], 1ULL);

	for (uint64_t column = 0; column < column_count; column++) {
		const auto input_idx = column * value_stride + row;
		if (!value_validity || value_validity[input_idx] != 0) {
			const auto output_idx = column * group_count + group;
			AtomicAddDouble(&sum_out[output_idx], values[input_idx]);
			atomicAdd(&count_out[output_idx], 1ULL);
		}
	}
}

__global__ void DuckDBGpuFusedLatAggMultiKernel(const int64_t *grids, const double *values,
                                                const uint8_t *value_validity, uint64_t column_count, uint64_t count,
                                                int64_t grid_min, int64_t grid_max, const int32_t *grid_to_group,
                                                uint64_t build_size, uint64_t group_count, double *sum_out,
                                                unsigned long long *count_out,
                                                unsigned long long *row_count_out) {
	const auto row = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	DuckDBGpuFusedLatAggMultiRow(grids, values, value_validity, column_count, count, count, grid_min, grid_max,
	                             grid_to_group, build_size, group_count, sum_out, count_out, row_count_out, row);
}

__global__ void DuckDBGpuFusedLatAggMultiStridedKernel(
    const int64_t *grids, const double *values, const uint8_t *value_validity, uint64_t column_count,
    uint64_t value_stride, uint64_t count, int64_t grid_min, int64_t grid_max, const int32_t *grid_to_group,
    uint64_t build_size, uint64_t group_count, double *sum_out, unsigned long long *count_out,
    unsigned long long *row_count_out) {
	const auto row = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	DuckDBGpuFusedLatAggMultiRow(grids, values, value_validity, column_count, value_stride, count, grid_min,
	                             grid_max, grid_to_group, build_size, group_count, sum_out, count_out,
	                             row_count_out, row);
}

int CheckCuda(cudaError_t status, const char *step) {
	if (status == cudaSuccess) {
		return 0;
	}
	std::fprintf(stderr, "[duckdb gpu offload] CUDA %s failed: %s\n", step, cudaGetErrorString(status));
	return 1;
}

__device__ double AtomicAddDouble(double *address, double value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
	return atomicAdd(address, value);
#else
	auto address_as_ull = reinterpret_cast<unsigned long long *>(address);
	auto old = *address_as_ull;
	unsigned long long assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(value + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
#endif
}

__device__ double AtomicMinDouble(double *address, double value) {
	auto address_as_ull = reinterpret_cast<unsigned long long *>(address);
	auto old = *address_as_ull;
	unsigned long long assumed;
	do {
		assumed = old;
		const auto current = __longlong_as_double(assumed);
		if (current <= value) {
			break;
		}
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(value));
	} while (assumed != old);
	return __longlong_as_double(old);
}

__device__ double AtomicMaxDouble(double *address, double value) {
	auto address_as_ull = reinterpret_cast<unsigned long long *>(address);
	auto old = *address_as_ull;
	unsigned long long assumed;
	do {
		assumed = old;
		const auto current = __longlong_as_double(assumed);
		if (current >= value) {
			break;
		}
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(value));
	} while (assumed != old);
	return __longlong_as_double(old);
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

struct ManagedBuffer {
	void *ptr = nullptr;
	size_t capacity = 0;

	~ManagedBuffer() {
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
		if (CheckCuda(cudaMallocManaged(&ptr, bytes), step)) {
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

struct GroupBySumDoubleBuffers {
	DeviceBuffer addresses;
	DeviceBuffer values;
	DeviceBuffer validity;
	DeviceBuffer unique_addresses;
	DeviceBuffer sums;
	DeviceBuffer unique_count;
};

struct GroupByStatsDoubleBuffers {
	DeviceBuffer addresses;
	DeviceBuffer values;
	DeviceBuffer validity;
	DeviceBuffer unique_addresses;
	DeviceBuffer sums;
	DeviceBuffer counts;
	DeviceBuffer mins;
	DeviceBuffer maxs;
	DeviceBuffer unique_count;
};

struct GroupByDictStatsDoubleBuffers {
	MappedHostBuffer group_ids;
	MappedHostBuffer values;
	MappedHostBuffer validity;
	MappedHostBuffer sums;
	MappedHostBuffer counts;
	MappedHostBuffer row_counts;
	MappedHostBuffer mins;
	MappedHostBuffer maxs;
};

struct FusedLatAggBuffers {
	DeviceBuffer grids;
	DeviceBuffer values;
	DeviceBuffer value_validity;
	DeviceBuffer grid_to_group;
	DeviceBuffer sums;
	DeviceBuffer counts;
	DeviceBuffer row_counts;
};

struct MappedFusedLatAggBuffers {
	MappedHostBuffer grids;
	MappedHostBuffer values;
	MappedHostBuffer value_validity;
	MappedHostBuffer grid_to_group;
	DeviceBuffer sums;
	DeviceBuffer counts;
	DeviceBuffer row_counts;
};

struct FusedLatAggMultiBuffers {
	DeviceBuffer grids;
	DeviceBuffer values;
	DeviceBuffer value_validity;
	DeviceBuffer grid_to_group;
	DeviceBuffer sums;
	DeviceBuffer counts;
	DeviceBuffer row_counts;
};

struct MappedFusedLatAggMultiBuffers {
	MappedHostBuffer grids;
	MappedHostBuffer values;
	MappedHostBuffer value_validity;
	MappedHostBuffer grid_to_group;
	DeviceBuffer sums;
	DeviceBuffer counts;
	DeviceBuffer row_counts;
};

struct FusedLatAggPipelineSlot {
	DeviceBuffer grids;
	DeviceBuffer values;
	DeviceBuffer value_validity;
	DeviceBuffer grid_to_group;
	MappedHostBuffer mapped_grids;
	MappedHostBuffer mapped_values;
	MappedHostBuffer mapped_value_validity;
	MappedHostBuffer mapped_grid_to_group;
	ManagedBuffer managed_grids;
	ManagedBuffer managed_values;
	ManagedBuffer managed_value_validity;
	ManagedBuffer managed_grid_to_group;
	DeviceBuffer sums;
	DeviceBuffer counts;
	DeviceBuffer row_counts;
	cudaStream_t stream = nullptr;
	bool active = false;
	int memory_mode = 0;
	uint64_t count = 0;
	uint64_t group_count = 0;
	uint64_t build_size = 0;
	uint64_t column_count = 0;
	uint64_t value_stride = 0;
	int64_t grid_min = 0;
	int64_t grid_max = 0;

	~FusedLatAggPipelineSlot() {
		if (stream) {
			cudaStreamSynchronize(stream);
			cudaStreamDestroy(stream);
		}
	}

	int EnsureStream() {
		if (stream) {
			return 0;
		}
		return CheckCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create fused pipeline stream");
	}

	bool IsMapped() const {
		return memory_mode == 1;
	}

	bool IsManaged() const {
		return memory_mode == 2;
	}

	bool IsDevice() const {
		return memory_mode == 0;
	}
};

struct FusedLatAggPipelineState {
	explicit FusedLatAggPipelineState(uint32_t slot_count_p, int memory_mode_p)
	    : slot_count(slot_count_p), slots(new FusedLatAggPipelineSlot[slot_count_p]) {
		for (uint32_t slot = 0; slot < slot_count; slot++) {
			slots[slot].memory_mode = memory_mode_p;
		}
	}

	~FusedLatAggPipelineState() {
		for (uint32_t slot = 0; slot < slot_count; slot++) {
			if (slots[slot].stream) {
				cudaStreamSynchronize(slots[slot].stream);
			}
		}
		delete[] slots;
	}

	uint32_t slot_count;
	FusedLatAggPipelineSlot *slots;
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

extern "C" int duckdb_gpu_groupby_sum_double(const uint64_t *addresses, const double *values, const uint8_t *validity,
                                             uint64_t count, uint64_t *unique_addresses_out, double *sums_out,
                                             uint64_t *unique_count_out) {
	if (!addresses || !values || !validity || !unique_addresses_out || !sums_out || !unique_count_out) {
		return 1;
	}
	if (count == 0) {
		*unique_count_out = 0;
		return 0;
	}

	unsigned long long result_count = 0;
	thread_local GroupBySumDoubleBuffers buffers;

	const auto addresses_bytes = count * sizeof(uint64_t);
	const auto values_bytes = count * sizeof(double);
	const auto validity_bytes = count * sizeof(uint8_t);
	const auto unique_addresses_bytes = count * sizeof(uint64_t);
	const auto sums_bytes = count * sizeof(double);

	int error = 0;
	error |= buffers.addresses.Ensure(addresses_bytes, "resize groupby sum addresses");
	error |= buffers.values.Ensure(values_bytes, "resize groupby sum values");
	error |= buffers.validity.Ensure(validity_bytes, "resize groupby sum validity");
	error |= buffers.unique_addresses.Ensure(unique_addresses_bytes, "resize groupby sum unique addresses");
	error |= buffers.sums.Ensure(sums_bytes, "resize groupby sum output sums");
	error |= buffers.unique_count.Ensure(sizeof(unsigned long long), "resize groupby sum unique count");
	if (error) {
		return 1;
	}

	auto d_addresses = buffers.addresses.As<uint64_t>();
	auto d_values = buffers.values.As<double>();
	auto d_validity = buffers.validity.As<uint8_t>();
	auto d_unique_addresses = buffers.unique_addresses.As<uint64_t>();
	auto d_sums = buffers.sums.As<double>();
	auto d_unique_count = buffers.unique_count.As<unsigned long long>();

	error |= CheckCuda(cudaMemcpy(d_addresses, addresses, addresses_bytes, cudaMemcpyHostToDevice),
	                   "copy groupby sum addresses to device");
	error |= CheckCuda(cudaMemcpy(d_values, values, values_bytes, cudaMemcpyHostToDevice),
	                   "copy groupby sum values to device");
	error |= CheckCuda(cudaMemcpy(d_validity, validity, validity_bytes, cudaMemcpyHostToDevice),
	                   "copy groupby sum validity to device");
	error |= CheckCuda(cudaMemset(d_unique_count, 0, sizeof(unsigned long long)), "clear groupby sum unique count");
	if (error) {
		return 1;
	}

	{
		constexpr int THREADS_PER_BLOCK = 256;
		const auto blocks = static_cast<unsigned int>((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
		DuckDBGpuGroupBySumDoubleKernel<<<blocks, THREADS_PER_BLOCK>>>(d_addresses, d_values, d_validity, count,
		                                                               d_unique_addresses, d_sums, d_unique_count);
		error |= CheckCuda(cudaGetLastError(), "launch groupby sum double kernel");
		error |= CheckCuda(cudaMemcpy(&result_count, d_unique_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost),
		                   "copy groupby sum unique count to host");
	}
	if (error || result_count > count) {
		return 1;
	}

	error |= CheckCuda(cudaMemcpy(unique_addresses_out, d_unique_addresses, result_count * sizeof(uint64_t),
	                             cudaMemcpyDeviceToHost),
	                   "copy groupby sum unique addresses to host");
	error |= CheckCuda(cudaMemcpy(sums_out, d_sums, result_count * sizeof(double), cudaMemcpyDeviceToHost),
	                   "copy groupby sums to host");
	*unique_count_out = static_cast<uint64_t>(result_count);
	return error ? 1 : 0;
}

extern "C" int duckdb_gpu_groupby_stats_double(const uint64_t *addresses, const double *values, const uint8_t *validity,
                                               uint64_t count, uint64_t *unique_addresses_out, double *sums_out,
                                               uint64_t *counts_out, double *mins_out, double *maxs_out,
                                               uint64_t *unique_count_out) {
	if (!addresses || !values || !validity || !unique_addresses_out || !sums_out || !counts_out || !mins_out ||
	    !maxs_out || !unique_count_out) {
		return 1;
	}
	if (count == 0) {
		*unique_count_out = 0;
		return 0;
	}

	unsigned long long result_count = 0;
	thread_local GroupByStatsDoubleBuffers buffers;

	const auto addresses_bytes = count * sizeof(uint64_t);
	const auto values_bytes = count * sizeof(double);
	const auto validity_bytes = count * sizeof(uint8_t);
	const auto unique_addresses_bytes = count * sizeof(uint64_t);
	const auto sums_bytes = count * sizeof(double);
	const auto counts_bytes = count * sizeof(unsigned long long);
	const auto mins_bytes = count * sizeof(double);
	const auto maxs_bytes = count * sizeof(double);

	int error = 0;
	error |= buffers.addresses.Ensure(addresses_bytes, "resize groupby stats addresses");
	error |= buffers.values.Ensure(values_bytes, "resize groupby stats values");
	error |= buffers.validity.Ensure(validity_bytes, "resize groupby stats validity");
	error |= buffers.unique_addresses.Ensure(unique_addresses_bytes, "resize groupby stats unique addresses");
	error |= buffers.sums.Ensure(sums_bytes, "resize groupby stats sums");
	error |= buffers.counts.Ensure(counts_bytes, "resize groupby stats counts");
	error |= buffers.mins.Ensure(mins_bytes, "resize groupby stats mins");
	error |= buffers.maxs.Ensure(maxs_bytes, "resize groupby stats maxs");
	error |= buffers.unique_count.Ensure(sizeof(unsigned long long), "resize groupby stats unique count");
	if (error) {
		return 1;
	}

	auto d_addresses = buffers.addresses.As<uint64_t>();
	auto d_values = buffers.values.As<double>();
	auto d_validity = buffers.validity.As<uint8_t>();
	auto d_unique_addresses = buffers.unique_addresses.As<uint64_t>();
	auto d_sums = buffers.sums.As<double>();
	auto d_counts = buffers.counts.As<unsigned long long>();
	auto d_mins = buffers.mins.As<double>();
	auto d_maxs = buffers.maxs.As<double>();
	auto d_unique_count = buffers.unique_count.As<unsigned long long>();

	error |= CheckCuda(cudaMemcpy(d_addresses, addresses, addresses_bytes, cudaMemcpyHostToDevice),
	                   "copy groupby stats addresses to device");
	error |= CheckCuda(cudaMemcpy(d_values, values, values_bytes, cudaMemcpyHostToDevice),
	                   "copy groupby stats values to device");
	error |= CheckCuda(cudaMemcpy(d_validity, validity, validity_bytes, cudaMemcpyHostToDevice),
	                   "copy groupby stats validity to device");
	error |= CheckCuda(cudaMemset(d_unique_count, 0, sizeof(unsigned long long)), "clear groupby stats unique count");
	if (error) {
		return 1;
	}

	{
		constexpr int THREADS_PER_BLOCK = 256;
		const auto blocks = static_cast<unsigned int>((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
		DuckDBGpuGroupByStatsDoubleKernel<<<blocks, THREADS_PER_BLOCK>>>(d_addresses, d_values, d_validity, count,
		                                                                 d_unique_addresses, d_sums, d_counts,
		                                                                 d_mins, d_maxs, d_unique_count);
		error |= CheckCuda(cudaGetLastError(), "launch groupby stats double kernel");
		error |= CheckCuda(cudaMemcpy(&result_count, d_unique_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost),
		                   "copy groupby stats unique count to host");
	}
	if (error || result_count > count) {
		return 1;
	}

	error |= CheckCuda(cudaMemcpy(unique_addresses_out, d_unique_addresses, result_count * sizeof(uint64_t),
	                             cudaMemcpyDeviceToHost),
	                   "copy groupby stats unique addresses to host");
	error |= CheckCuda(cudaMemcpy(sums_out, d_sums, result_count * sizeof(double), cudaMemcpyDeviceToHost),
	                   "copy groupby stats sums to host");
	error |= CheckCuda(cudaMemcpy(counts_out, d_counts, result_count * sizeof(unsigned long long),
	                             cudaMemcpyDeviceToHost),
	                   "copy groupby stats counts to host");
	error |= CheckCuda(cudaMemcpy(mins_out, d_mins, result_count * sizeof(double), cudaMemcpyDeviceToHost),
	                   "copy groupby stats mins to host");
	error |= CheckCuda(cudaMemcpy(maxs_out, d_maxs, result_count * sizeof(double), cudaMemcpyDeviceToHost),
	                   "copy groupby stats maxs to host");
	*unique_count_out = static_cast<uint64_t>(result_count);
	return error ? 1 : 0;
}

extern "C" int duckdb_gpu_groupby_dict_stats_double(const uint32_t *group_ids, const double *values,
                                                    const uint8_t *validity, uint64_t count, uint64_t group_count,
                                                    double *sums_out, uint64_t *counts_out, uint64_t *row_counts_out,
                                                    double *mins_out, double *maxs_out) {
	if (!group_ids || !values || !validity || !sums_out || !counts_out || !row_counts_out || !mins_out || !maxs_out) {
		return 1;
	}
	if (group_count == 0) {
		return 0;
	}

	thread_local GroupByDictStatsDoubleBuffers buffers;
	const auto group_ids_bytes = count * sizeof(uint32_t);
	const auto values_bytes = count * sizeof(double);
	const auto validity_bytes = count * sizeof(uint8_t);
	const auto sums_bytes = group_count * sizeof(double);
	const auto counts_bytes = group_count * sizeof(unsigned long long);
	const auto mins_bytes = group_count * sizeof(double);
	const auto maxs_bytes = group_count * sizeof(double);

	int error = 0;
	error |= buffers.group_ids.Ensure(group_ids_bytes, "resize dict stats group ids");
	error |= buffers.values.Ensure(values_bytes, "resize dict stats values");
	error |= buffers.validity.Ensure(validity_bytes, "resize dict stats validity");
	error |= buffers.sums.Ensure(sums_bytes, "resize dict stats sums");
	error |= buffers.counts.Ensure(counts_bytes, "resize dict stats counts");
	error |= buffers.row_counts.Ensure(counts_bytes, "resize dict stats row counts");
	error |= buffers.mins.Ensure(mins_bytes, "resize dict stats mins");
	error |= buffers.maxs.Ensure(maxs_bytes, "resize dict stats maxs");
	if (error) {
		return 1;
	}

	auto h_group_ids = buffers.group_ids.HostAs<uint32_t>();
	auto h_values = buffers.values.HostAs<double>();
	auto h_validity = buffers.validity.HostAs<uint8_t>();
	auto h_sums = buffers.sums.HostAs<double>();
	auto h_counts = buffers.counts.HostAs<unsigned long long>();
	auto h_row_counts = buffers.row_counts.HostAs<unsigned long long>();
	auto h_mins = buffers.mins.HostAs<double>();
	auto h_maxs = buffers.maxs.HostAs<double>();

	auto d_group_ids = buffers.group_ids.DeviceAs<uint32_t>();
	auto d_values = buffers.values.DeviceAs<double>();
	auto d_validity = buffers.validity.DeviceAs<uint8_t>();
	auto d_sums = buffers.sums.DeviceAs<double>();
	auto d_counts = buffers.counts.DeviceAs<unsigned long long>();
	auto d_row_counts = buffers.row_counts.DeviceAs<unsigned long long>();
	auto d_mins = buffers.mins.DeviceAs<double>();
	auto d_maxs = buffers.maxs.DeviceAs<double>();

	std::memcpy(h_group_ids, group_ids, group_ids_bytes);
	std::memcpy(h_values, values, values_bytes);
	std::memcpy(h_validity, validity, validity_bytes);

	{
		constexpr int THREADS_PER_BLOCK = 256;
		auto blocks = static_cast<unsigned int>((group_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
		DuckDBGpuInitDictStatsDoubleKernel<<<blocks, THREADS_PER_BLOCK>>>(group_count, d_sums, d_counts, d_row_counts,
		                                                                  d_mins, d_maxs);
		error |= CheckCuda(cudaGetLastError(), "launch dict stats init kernel");
		blocks = static_cast<unsigned int>((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
		DuckDBGpuGroupByDictStatsDoubleKernel<<<blocks, THREADS_PER_BLOCK>>>(d_group_ids, d_values, d_validity, count,
		                                                                     group_count, d_sums, d_counts,
		                                                                     d_row_counts, d_mins, d_maxs);
		error |= CheckCuda(cudaGetLastError(), "launch dict stats kernel");
		error |= CheckCuda(cudaDeviceSynchronize(), "sync dict stats kernel");
	}
	if (error) {
		return 1;
	}

	std::memcpy(sums_out, h_sums, sums_bytes);
	std::memcpy(counts_out, h_counts, counts_bytes);
	std::memcpy(row_counts_out, h_row_counts, counts_bytes);
	std::memcpy(mins_out, h_mins, mins_bytes);
	std::memcpy(maxs_out, h_maxs, maxs_bytes);
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

	auto d_grids = buffers.grids.As<int64_t>();
	auto d_values = buffers.values.As<double>();
	auto d_value_validity = buffers.value_validity.As<uint8_t>();
	auto d_grid_to_group = buffers.grid_to_group.As<int32_t>();
	auto d_sums = buffers.sums.As<double>();
	auto d_counts = buffers.counts.As<unsigned long long>();
	auto d_row_counts = buffers.row_counts.As<unsigned long long>();

	error |= CheckCuda(cudaMemcpy(d_grids, grids, grid_bytes, cudaMemcpyHostToDevice), "copy fused grids to device");
	error |= CheckCuda(cudaMemcpy(d_values, values, value_bytes, cudaMemcpyHostToDevice), "copy fused values to device");
	if (value_validity) {
		error |= CheckCuda(cudaMemcpy(d_value_validity, value_validity, validity_bytes, cudaMemcpyHostToDevice),
		                   "copy fused value validity to device");
	} else {
		error |= CheckCuda(cudaMemset(d_value_validity, 1, validity_bytes), "set fused value validity");
	}
	error |= CheckCuda(cudaMemcpy(d_grid_to_group, grid_to_group, build_bytes, cudaMemcpyHostToDevice),
	                   "copy fused grid to group to device");

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

extern "C" int duckdb_gpu_fused_lat_agg_i64_double_mapped(const int64_t *grids, const double *values,
                                                          const uint8_t *value_validity, uint64_t count,
                                                          int64_t grid_min, int64_t grid_max,
                                                          const int32_t *grid_to_group, uint64_t build_size,
                                                          uint64_t group_count, double *sum_out, uint64_t *count_out,
                                                          uint64_t *row_count_out) {
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

	thread_local MappedFusedLatAggBuffers buffers;
	const auto grid_bytes = count * sizeof(int64_t);
	const auto value_bytes = count * sizeof(double);
	const auto validity_bytes = count * sizeof(uint8_t);
	const auto build_bytes = build_size * sizeof(int32_t);
	const auto sum_bytes = group_count * sizeof(double);
	const auto count_bytes = group_count * sizeof(unsigned long long);

	int error = 0;
	error |= buffers.grids.Ensure(grid_bytes, "resize mapped fused grids");
	error |= buffers.values.Ensure(value_bytes, "resize mapped fused values");
	error |= buffers.value_validity.Ensure(validity_bytes, "resize mapped fused value validity");
	error |= buffers.grid_to_group.Ensure(build_bytes, "resize mapped fused grid to group");
	error |= buffers.sums.Ensure(sum_bytes, "resize mapped fused sums");
	error |= buffers.counts.Ensure(count_bytes, "resize mapped fused counts");
	error |= buffers.row_counts.Ensure(count_bytes, "resize mapped fused row counts");
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

	error |= CheckCuda(cudaMemset(d_sums, 0, sum_bytes), "clear mapped fused sums");
	error |= CheckCuda(cudaMemset(d_counts, 0, count_bytes), "clear mapped fused counts");
	error |= CheckCuda(cudaMemset(d_row_counts, 0, count_bytes), "clear mapped fused row counts");
	if (error) {
		return 1;
	}

	{
		constexpr int THREADS_PER_BLOCK = 256;
		const auto blocks = static_cast<unsigned int>((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
		DuckDBGpuFusedLatAggKernel<<<blocks, THREADS_PER_BLOCK>>>(d_grids, d_values, d_value_validity, count, grid_min,
		                                                          grid_max, d_grid_to_group, build_size, d_sums,
		                                                          d_counts, d_row_counts);
		error |= CheckCuda(cudaGetLastError(), "launch mapped fused lat aggregate kernel");
		error |= CheckCuda(cudaDeviceSynchronize(), "synchronize mapped fused lat aggregate kernel");
	}
	if (error) {
		return 1;
	}

	error |= CheckCuda(cudaMemcpy(sum_out, d_sums, sum_bytes, cudaMemcpyDeviceToHost),
	                   "copy mapped fused sums to host");
	error |= CheckCuda(cudaMemcpy(count_out, d_counts, count_bytes, cudaMemcpyDeviceToHost),
	                   "copy mapped fused counts to host");
	error |= CheckCuda(cudaMemcpy(row_count_out, d_row_counts, count_bytes, cudaMemcpyDeviceToHost),
	                   "copy mapped fused row counts to host");
	return error ? 1 : 0;
}

extern "C" int duckdb_gpu_fused_lat_agg_multi_i64_double_strided(
    const int64_t *grids, const double *values, const uint8_t *value_validity, uint64_t column_count,
    uint64_t value_stride, uint64_t count, int64_t grid_min, int64_t grid_max, const int32_t *grid_to_group,
    uint64_t build_size, uint64_t group_count, double *sum_out, uint64_t *count_out, uint64_t *row_count_out) {
	if (!grids || !values || !grid_to_group || !sum_out || !count_out || !row_count_out || column_count == 0) {
		return 1;
	}
	if (group_count == 0) {
		return 0;
	}
	const auto output_count = column_count * group_count;
	if (count == 0) {
		std::memset(sum_out, 0, output_count * sizeof(double));
		std::memset(count_out, 0, output_count * sizeof(uint64_t));
		std::memset(row_count_out, 0, group_count * sizeof(uint64_t));
		return 0;
	}
	if (grid_max < grid_min || build_size == 0) {
		return 1;
	}
	if (value_stride < count) {
		return 1;
	}

	thread_local FusedLatAggMultiBuffers buffers;
	const auto grid_bytes = count * sizeof(int64_t);
	const auto value_bytes = column_count * value_stride * sizeof(double);
	const auto validity_bytes = column_count * value_stride * sizeof(uint8_t);
	const bool assume_all_valid = !value_validity || AssumePayloadAllValid();
	const auto build_bytes = build_size * sizeof(int32_t);
	const auto sum_bytes = output_count * sizeof(double);
	const auto output_count_bytes = output_count * sizeof(unsigned long long);
	const auto row_count_bytes = group_count * sizeof(unsigned long long);

	int error = 0;
	error |= buffers.grids.Ensure(grid_bytes, "resize multi fused grids");
	error |= buffers.values.Ensure(value_bytes, "resize multi fused values");
	if (!assume_all_valid) {
		error |= buffers.value_validity.Ensure(validity_bytes, "resize multi fused value validity");
	}
	error |= buffers.grid_to_group.Ensure(build_bytes, "resize multi fused grid to group");
	error |= buffers.sums.Ensure(sum_bytes, "resize multi fused sums");
	error |= buffers.counts.Ensure(output_count_bytes, "resize multi fused counts");
	error |= buffers.row_counts.Ensure(row_count_bytes, "resize multi fused row counts");
	if (error) {
		return 1;
	}

	auto d_grids = buffers.grids.As<int64_t>();
	auto d_values = buffers.values.As<double>();
	uint8_t *d_value_validity = nullptr;
	auto d_grid_to_group = buffers.grid_to_group.As<int32_t>();
	auto d_sums = buffers.sums.As<double>();
	auto d_counts = buffers.counts.As<unsigned long long>();
	auto d_row_counts = buffers.row_counts.As<unsigned long long>();

	error |= CheckCuda(cudaMemcpy(d_grids, grids, grid_bytes, cudaMemcpyHostToDevice),
	                   "copy multi fused grids to device");
	error |= CheckCuda(cudaMemcpy(d_values, values, value_bytes, cudaMemcpyHostToDevice),
	                   "copy multi fused values to device");
	if (!assume_all_valid) {
		d_value_validity = buffers.value_validity.As<uint8_t>();
		error |= CheckCuda(cudaMemcpy(d_value_validity, value_validity, validity_bytes, cudaMemcpyHostToDevice),
		                   "copy multi fused value validity to device");
	}
	error |= CheckCuda(cudaMemcpy(d_grid_to_group, grid_to_group, build_bytes, cudaMemcpyHostToDevice),
	                   "copy multi fused grid to group to device");
	error |= CheckCuda(cudaMemset(d_sums, 0, sum_bytes), "clear multi fused sums");
	error |= CheckCuda(cudaMemset(d_counts, 0, output_count_bytes), "clear multi fused counts");
	error |= CheckCuda(cudaMemset(d_row_counts, 0, row_count_bytes), "clear multi fused row counts");
	if (error) {
		return 1;
	}

	constexpr int THREADS_PER_BLOCK = 256;
	const auto blocks = static_cast<unsigned int>((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
	DuckDBGpuFusedLatAggMultiStridedKernel<<<blocks, THREADS_PER_BLOCK>>>(
	    d_grids, d_values, d_value_validity, column_count, value_stride, count, grid_min, grid_max, d_grid_to_group,
	    build_size, group_count, d_sums, d_counts, d_row_counts);
	error |= CheckCuda(cudaGetLastError(), "launch multi fused lat aggregate kernel");
	error |= CheckCuda(cudaDeviceSynchronize(), "synchronize multi fused lat aggregate kernel");
	if (error) {
		return 1;
	}

	error |= CheckCuda(cudaMemcpy(sum_out, d_sums, sum_bytes, cudaMemcpyDeviceToHost),
	                   "copy multi fused sums to host");
	error |= CheckCuda(cudaMemcpy(count_out, d_counts, output_count_bytes, cudaMemcpyDeviceToHost),
	                   "copy multi fused counts to host");
	error |= CheckCuda(cudaMemcpy(row_count_out, d_row_counts, row_count_bytes, cudaMemcpyDeviceToHost),
	                   "copy multi fused row counts to host");
	return error ? 1 : 0;
}

extern "C" int duckdb_gpu_fused_lat_agg_multi_i64_double(
    const int64_t *grids, const double *values, const uint8_t *value_validity, uint64_t column_count, uint64_t count,
    int64_t grid_min, int64_t grid_max, const int32_t *grid_to_group, uint64_t build_size, uint64_t group_count,
    double *sum_out, uint64_t *count_out, uint64_t *row_count_out) {
	return duckdb_gpu_fused_lat_agg_multi_i64_double_strided(
	    grids, values, value_validity, column_count, count, count, grid_min, grid_max, grid_to_group, build_size,
	    group_count, sum_out, count_out, row_count_out);
}

extern "C" int duckdb_gpu_fused_lat_agg_multi_i64_double_strided_mapped(
    const int64_t *grids, const double *values, const uint8_t *value_validity, uint64_t column_count,
    uint64_t value_stride, uint64_t count, int64_t grid_min, int64_t grid_max, const int32_t *grid_to_group,
    uint64_t build_size, uint64_t group_count, double *sum_out, uint64_t *count_out, uint64_t *row_count_out) {
	if (!grids || !values || !grid_to_group || !sum_out || !count_out || !row_count_out || column_count == 0) {
		return 1;
	}
	if (group_count == 0) {
		return 0;
	}
	const auto output_count = column_count * group_count;
	if (count == 0) {
		std::memset(sum_out, 0, output_count * sizeof(double));
		std::memset(count_out, 0, output_count * sizeof(uint64_t));
		std::memset(row_count_out, 0, group_count * sizeof(uint64_t));
		return 0;
	}
	if (grid_max < grid_min || build_size == 0) {
		return 1;
	}
	if (value_stride < count) {
		return 1;
	}

	thread_local MappedFusedLatAggMultiBuffers buffers;
	const auto grid_bytes = count * sizeof(int64_t);
	const auto value_bytes = column_count * value_stride * sizeof(double);
	const auto validity_bytes = column_count * value_stride * sizeof(uint8_t);
	const bool assume_all_valid = !value_validity || AssumePayloadAllValid();
	const auto build_bytes = build_size * sizeof(int32_t);
	const auto sum_bytes = output_count * sizeof(double);
	const auto output_count_bytes = output_count * sizeof(unsigned long long);
	const auto row_count_bytes = group_count * sizeof(unsigned long long);

	int error = 0;
	error |= buffers.grids.Ensure(grid_bytes, "resize mapped multi fused grids");
	error |= buffers.values.Ensure(value_bytes, "resize mapped multi fused values");
	if (!assume_all_valid) {
		error |= buffers.value_validity.Ensure(validity_bytes, "resize mapped multi fused value validity");
	}
	error |= buffers.grid_to_group.Ensure(build_bytes, "resize mapped multi fused grid to group");
	error |= buffers.sums.Ensure(sum_bytes, "resize mapped multi fused sums");
	error |= buffers.counts.Ensure(output_count_bytes, "resize mapped multi fused counts");
	error |= buffers.row_counts.Ensure(row_count_bytes, "resize mapped multi fused row counts");
	if (error) {
		return 1;
	}

	auto h_grids = buffers.grids.HostAs<int64_t>();
	auto h_values = buffers.values.HostAs<double>();
	auto h_grid_to_group = buffers.grid_to_group.HostAs<int32_t>();
	auto d_grids = buffers.grids.DeviceAs<int64_t>();
	auto d_values = buffers.values.DeviceAs<double>();
	uint8_t *d_value_validity = nullptr;
	auto d_grid_to_group = buffers.grid_to_group.DeviceAs<int32_t>();
	auto d_sums = buffers.sums.As<double>();
	auto d_counts = buffers.counts.As<unsigned long long>();
	auto d_row_counts = buffers.row_counts.As<unsigned long long>();

	std::memcpy(h_grids, grids, grid_bytes);
	std::memcpy(h_values, values, value_bytes);
	if (!assume_all_valid) {
		auto h_value_validity = buffers.value_validity.HostAs<uint8_t>();
		std::memcpy(h_value_validity, value_validity, validity_bytes);
		d_value_validity = buffers.value_validity.DeviceAs<uint8_t>();
	}
	std::memcpy(h_grid_to_group, grid_to_group, build_bytes);

	error |= CheckCuda(cudaMemset(d_sums, 0, sum_bytes), "clear mapped multi fused sums");
	error |= CheckCuda(cudaMemset(d_counts, 0, output_count_bytes), "clear mapped multi fused counts");
	error |= CheckCuda(cudaMemset(d_row_counts, 0, row_count_bytes), "clear mapped multi fused row counts");
	if (error) {
		return 1;
	}

	constexpr int THREADS_PER_BLOCK = 256;
	const auto blocks = static_cast<unsigned int>((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
	DuckDBGpuFusedLatAggMultiStridedKernel<<<blocks, THREADS_PER_BLOCK>>>(
	    d_grids, d_values, d_value_validity, column_count, value_stride, count, grid_min, grid_max, d_grid_to_group,
	    build_size, group_count, d_sums, d_counts, d_row_counts);
	error |= CheckCuda(cudaGetLastError(), "launch mapped multi fused lat aggregate kernel");
	error |= CheckCuda(cudaDeviceSynchronize(), "synchronize mapped multi fused lat aggregate kernel");
	if (error) {
		return 1;
	}

	error |= CheckCuda(cudaMemcpy(sum_out, d_sums, sum_bytes, cudaMemcpyDeviceToHost),
	                   "copy mapped multi fused sums to host");
	error |= CheckCuda(cudaMemcpy(count_out, d_counts, output_count_bytes, cudaMemcpyDeviceToHost),
	                   "copy mapped multi fused counts to host");
	error |= CheckCuda(cudaMemcpy(row_count_out, d_row_counts, row_count_bytes, cudaMemcpyDeviceToHost),
	                   "copy mapped multi fused row counts to host");
	return error ? 1 : 0;
}

extern "C" int duckdb_gpu_fused_lat_agg_multi_i64_double_mapped(
    const int64_t *grids, const double *values, const uint8_t *value_validity, uint64_t column_count, uint64_t count,
    int64_t grid_min, int64_t grid_max, const int32_t *grid_to_group, uint64_t build_size, uint64_t group_count,
    double *sum_out, uint64_t *count_out, uint64_t *row_count_out) {
	return duckdb_gpu_fused_lat_agg_multi_i64_double_strided_mapped(
	    grids, values, value_validity, column_count, count, count, grid_min, grid_max, grid_to_group, build_size,
	    group_count, sum_out, count_out, row_count_out);
}

extern "C" void *duckdb_gpu_fused_lat_agg_pipeline_create(uint32_t slot_count, int mapped) {
	if (slot_count == 0) {
		slot_count = 2;
	}
	auto memory_mode = mapped;
	if (memory_mode < 0 || memory_mode > 2) {
		memory_mode = mapped != 0 ? 1 : 0;
	}
	auto state = new FusedLatAggPipelineState(slot_count, memory_mode);
	for (uint32_t slot = 0; slot < slot_count; slot++) {
		if (state->slots[slot].EnsureStream()) {
			delete state;
			return nullptr;
		}
	}
	return state;
}

extern "C" int duckdb_gpu_fused_lat_agg_pipeline_submit_i64_double(
    void *handle, uint32_t slot_idx, const int64_t *grids, const double *values, const uint8_t *value_validity,
    uint64_t count, int64_t grid_min, int64_t grid_max, const int32_t *grid_to_group, uint64_t build_size,
    uint64_t group_count) {
	if (!handle || !grids || !values || !grid_to_group || group_count == 0) {
		return 1;
	}
	if (grid_max < grid_min || build_size == 0) {
		return 1;
	}

	auto state = reinterpret_cast<FusedLatAggPipelineState *>(handle);
	if (slot_idx >= state->slot_count) {
		return 1;
	}
	auto &slot = state->slots[slot_idx];
	if (slot.EnsureStream()) {
		return 1;
	}
	if (slot.active) {
		if (CheckCuda(cudaStreamSynchronize(slot.stream), "wait active fused pipeline slot before reuse")) {
			return 1;
		}
		slot.active = false;
	}

	const auto grid_bytes = count * sizeof(int64_t);
	const auto value_bytes = count * sizeof(double);
	const auto validity_bytes = count * sizeof(uint8_t);
	const auto build_bytes = build_size * sizeof(int32_t);
	const auto sum_bytes = group_count * sizeof(double);
	const auto count_bytes = group_count * sizeof(unsigned long long);

	int error = 0;
	error |= slot.sums.Ensure(sum_bytes, "resize pipeline fused sums");
	error |= slot.counts.Ensure(count_bytes, "resize pipeline fused counts");
	error |= slot.row_counts.Ensure(count_bytes, "resize pipeline fused row counts");
	if (slot.IsMapped()) {
		error |= slot.mapped_grids.Ensure(grid_bytes, "resize mapped pipeline fused grids");
		error |= slot.mapped_values.Ensure(value_bytes, "resize mapped pipeline fused values");
		error |= slot.mapped_value_validity.Ensure(validity_bytes, "resize mapped pipeline fused value validity");
		error |= slot.mapped_grid_to_group.Ensure(build_bytes, "resize mapped pipeline fused grid to group");
	} else if (slot.IsManaged()) {
		error |= slot.managed_grids.Ensure(grid_bytes, "resize managed pipeline fused grids");
		error |= slot.managed_values.Ensure(value_bytes, "resize managed pipeline fused values");
		error |= slot.managed_value_validity.Ensure(validity_bytes, "resize managed pipeline fused value validity");
		error |= slot.managed_grid_to_group.Ensure(build_bytes, "resize managed pipeline fused grid to group");
	} else {
		error |= slot.grids.Ensure(grid_bytes, "resize pipeline fused grids");
		error |= slot.values.Ensure(value_bytes, "resize pipeline fused values");
		error |= slot.value_validity.Ensure(validity_bytes, "resize pipeline fused value validity");
		error |= slot.grid_to_group.Ensure(build_bytes, "resize pipeline fused grid to group");
	}
	if (error) {
		return 1;
	}

	int64_t *d_grids = nullptr;
	double *d_values = nullptr;
	uint8_t *d_value_validity = nullptr;
	int32_t *d_grid_to_group = nullptr;
	auto d_sums = slot.sums.As<double>();
	auto d_counts = slot.counts.As<unsigned long long>();
	auto d_row_counts = slot.row_counts.As<unsigned long long>();

	if (slot.IsMapped()) {
		auto h_grids = slot.mapped_grids.HostAs<int64_t>();
		auto h_values = slot.mapped_values.HostAs<double>();
		auto h_value_validity = slot.mapped_value_validity.HostAs<uint8_t>();
		auto h_grid_to_group = slot.mapped_grid_to_group.HostAs<int32_t>();
		std::memcpy(h_grids, grids, grid_bytes);
		std::memcpy(h_values, values, value_bytes);
		if (value_validity) {
			std::memcpy(h_value_validity, value_validity, validity_bytes);
		} else {
			std::memset(h_value_validity, 1, validity_bytes);
		}
		std::memcpy(h_grid_to_group, grid_to_group, build_bytes);

		d_grids = slot.mapped_grids.DeviceAs<int64_t>();
		d_values = slot.mapped_values.DeviceAs<double>();
		d_value_validity = slot.mapped_value_validity.DeviceAs<uint8_t>();
		d_grid_to_group = slot.mapped_grid_to_group.DeviceAs<int32_t>();
	} else if (slot.IsManaged()) {
		d_grids = slot.managed_grids.As<int64_t>();
		d_values = slot.managed_values.As<double>();
		d_value_validity = slot.managed_value_validity.As<uint8_t>();
		d_grid_to_group = slot.managed_grid_to_group.As<int32_t>();
		std::memcpy(d_grids, grids, grid_bytes);
		std::memcpy(d_values, values, value_bytes);
		if (value_validity) {
			std::memcpy(d_value_validity, value_validity, validity_bytes);
		} else {
			std::memset(d_value_validity, 1, validity_bytes);
		}
		std::memcpy(d_grid_to_group, grid_to_group, build_bytes);
	} else {
		d_grids = slot.grids.As<int64_t>();
		d_values = slot.values.As<double>();
		d_value_validity = slot.value_validity.As<uint8_t>();
		d_grid_to_group = slot.grid_to_group.As<int32_t>();
		error |= CheckCuda(cudaMemcpyAsync(d_grids, grids, grid_bytes, cudaMemcpyHostToDevice, slot.stream),
		                   "async copy pipeline fused grids to device");
		error |= CheckCuda(cudaMemcpyAsync(d_values, values, value_bytes, cudaMemcpyHostToDevice, slot.stream),
		                   "async copy pipeline fused values to device");
		if (value_validity) {
			error |= CheckCuda(cudaMemcpyAsync(d_value_validity, value_validity, validity_bytes,
			                                  cudaMemcpyHostToDevice, slot.stream),
			                   "async copy pipeline fused value validity to device");
		} else {
			error |= CheckCuda(cudaMemsetAsync(d_value_validity, 1, validity_bytes, slot.stream),
			                   "async set pipeline fused value validity");
		}
		error |= CheckCuda(cudaMemcpyAsync(d_grid_to_group, grid_to_group, build_bytes, cudaMemcpyHostToDevice,
		                                  slot.stream),
		                   "async copy pipeline fused grid to group to device");
	}

	error |= CheckCuda(cudaMemsetAsync(d_sums, 0, sum_bytes, slot.stream), "async clear pipeline fused sums");
	error |= CheckCuda(cudaMemsetAsync(d_counts, 0, count_bytes, slot.stream), "async clear pipeline fused counts");
	error |=
	    CheckCuda(cudaMemsetAsync(d_row_counts, 0, count_bytes, slot.stream), "async clear pipeline fused row counts");
	if (error) {
		return 1;
	}

	constexpr int THREADS_PER_BLOCK = 256;
	const auto blocks = static_cast<unsigned int>((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
	DuckDBGpuFusedLatAggKernel<<<blocks, THREADS_PER_BLOCK, 0, slot.stream>>>(
	    d_grids, d_values, d_value_validity, count, grid_min, grid_max, d_grid_to_group, build_size, d_sums, d_counts,
	    d_row_counts);
	error |= CheckCuda(cudaGetLastError(), "launch pipeline fused lat aggregate kernel");
	if (error) {
		return 1;
	}

	slot.count = count;
	slot.group_count = group_count;
	slot.active = true;
	return 0;
}

extern "C" int duckdb_gpu_fused_lat_agg_pipeline_reset_i64_double(void *handle, uint32_t slot_idx,
                                                                   int64_t grid_min, int64_t grid_max,
                                                                   const int32_t *grid_to_group,
                                                                   uint64_t build_size, uint64_t group_count) {
	if (!handle || !grid_to_group || group_count == 0) {
		return 1;
	}
	if (grid_max < grid_min || build_size == 0) {
		return 1;
	}

	auto state = reinterpret_cast<FusedLatAggPipelineState *>(handle);
	if (slot_idx >= state->slot_count) {
		return 1;
	}
	auto &slot = state->slots[slot_idx];
	if (slot.EnsureStream()) {
		return 1;
	}
	if (slot.active) {
		if (CheckCuda(cudaStreamSynchronize(slot.stream), "wait fused pipeline slot before accumulator reset")) {
			return 1;
		}
		slot.active = false;
	}

	const auto build_bytes = build_size * sizeof(int32_t);
	const auto sum_bytes = group_count * sizeof(double);
	const auto count_bytes = group_count * sizeof(unsigned long long);

	int error = 0;
	error |= slot.sums.Ensure(sum_bytes, "resize pipeline accumulator sums");
	error |= slot.counts.Ensure(count_bytes, "resize pipeline accumulator counts");
	error |= slot.row_counts.Ensure(count_bytes, "resize pipeline accumulator row counts");
	if (slot.IsMapped()) {
		error |= slot.mapped_grid_to_group.Ensure(build_bytes, "resize mapped pipeline accumulator grid to group");
	} else if (slot.IsManaged()) {
		error |= slot.managed_grid_to_group.Ensure(build_bytes, "resize managed pipeline accumulator grid to group");
	} else {
		error |= slot.grid_to_group.Ensure(build_bytes, "resize pipeline accumulator grid to group");
	}
	if (error) {
		return 1;
	}

	if (slot.IsMapped()) {
		std::memcpy(slot.mapped_grid_to_group.HostAs<int32_t>(), grid_to_group, build_bytes);
	} else if (slot.IsManaged()) {
		std::memcpy(slot.managed_grid_to_group.As<int32_t>(), grid_to_group, build_bytes);
	} else {
		error |= CheckCuda(cudaMemcpyAsync(slot.grid_to_group.As<int32_t>(), grid_to_group, build_bytes,
		                                  cudaMemcpyHostToDevice, slot.stream),
		                   "async copy pipeline accumulator grid to group to device");
	}
	error |= CheckCuda(cudaMemsetAsync(slot.sums.As<double>(), 0, sum_bytes, slot.stream),
	                   "async clear pipeline accumulator sums");
	error |= CheckCuda(cudaMemsetAsync(slot.counts.As<unsigned long long>(), 0, count_bytes, slot.stream),
	                   "async clear pipeline accumulator counts");
	error |= CheckCuda(cudaMemsetAsync(slot.row_counts.As<unsigned long long>(), 0, count_bytes, slot.stream),
	                   "async clear pipeline accumulator row counts");
	if (error) {
		return 1;
	}

	slot.count = 0;
	slot.group_count = group_count;
	slot.build_size = build_size;
	slot.grid_min = grid_min;
	slot.grid_max = grid_max;
	slot.active = true;
	return 0;
}

extern "C" int duckdb_gpu_fused_lat_agg_pipeline_submit_accumulate_i64_double(
    void *handle, uint32_t slot_idx, const int64_t *grids, const double *values, const uint8_t *value_validity,
    uint64_t count) {
	if (!handle || !grids || !values) {
		return 1;
	}
	if (count == 0) {
		return 0;
	}

	auto state = reinterpret_cast<FusedLatAggPipelineState *>(handle);
	if (slot_idx >= state->slot_count) {
		return 1;
	}
	auto &slot = state->slots[slot_idx];
	if (slot.EnsureStream()) {
		return 1;
	}
	if (slot.group_count == 0 || slot.build_size == 0) {
		return 1;
	}

	const auto grid_bytes = count * sizeof(int64_t);
	const auto value_bytes = count * sizeof(double);
	const auto validity_bytes = count * sizeof(uint8_t);

	int error = 0;
	if (slot.IsMapped()) {
		if (slot.active) {
			error |= CheckCuda(cudaStreamSynchronize(slot.stream),
			                   "sync mapped pipeline accumulator before input overwrite");
		}
		error |= slot.mapped_grids.Ensure(grid_bytes, "resize mapped pipeline accumulator grids");
		error |= slot.mapped_values.Ensure(value_bytes, "resize mapped pipeline accumulator values");
		error |= slot.mapped_value_validity.Ensure(validity_bytes, "resize mapped pipeline accumulator value validity");
		if (error) {
			return 1;
		}
		std::memcpy(slot.mapped_grids.HostAs<int64_t>(), grids, grid_bytes);
		std::memcpy(slot.mapped_values.HostAs<double>(), values, value_bytes);
		if (value_validity) {
			std::memcpy(slot.mapped_value_validity.HostAs<uint8_t>(), value_validity, validity_bytes);
		} else {
			std::memset(slot.mapped_value_validity.HostAs<uint8_t>(), 1, validity_bytes);
		}
	} else if (slot.IsManaged()) {
		if (slot.active) {
			error |= CheckCuda(cudaStreamSynchronize(slot.stream),
			                   "sync managed pipeline accumulator before input overwrite");
		}
		error |= slot.managed_grids.Ensure(grid_bytes, "resize managed pipeline accumulator grids");
		error |= slot.managed_values.Ensure(value_bytes, "resize managed pipeline accumulator values");
		error |= slot.managed_value_validity.Ensure(validity_bytes, "resize managed pipeline accumulator value validity");
		if (error) {
			return 1;
		}
		std::memcpy(slot.managed_grids.As<int64_t>(), grids, grid_bytes);
		std::memcpy(slot.managed_values.As<double>(), values, value_bytes);
		if (value_validity) {
			std::memcpy(slot.managed_value_validity.As<uint8_t>(), value_validity, validity_bytes);
		} else {
			std::memset(slot.managed_value_validity.As<uint8_t>(), 1, validity_bytes);
		}
	} else {
		error |= slot.grids.Ensure(grid_bytes, "resize pipeline accumulator grids");
		error |= slot.values.Ensure(value_bytes, "resize pipeline accumulator values");
		error |= slot.value_validity.Ensure(validity_bytes, "resize pipeline accumulator value validity");
		if (error) {
			return 1;
		}
		error |= CheckCuda(cudaMemcpyAsync(slot.grids.As<int64_t>(), grids, grid_bytes, cudaMemcpyHostToDevice,
		                                  slot.stream),
		                   "async copy pipeline accumulator grids to device");
		error |= CheckCuda(cudaMemcpyAsync(slot.values.As<double>(), values, value_bytes, cudaMemcpyHostToDevice,
		                                  slot.stream),
		                   "async copy pipeline accumulator values to device");
		if (value_validity) {
			error |= CheckCuda(cudaMemcpyAsync(slot.value_validity.As<uint8_t>(), value_validity, validity_bytes,
			                                  cudaMemcpyHostToDevice, slot.stream),
			                   "async copy pipeline accumulator value validity to device");
		} else {
			error |= CheckCuda(cudaMemsetAsync(slot.value_validity.As<uint8_t>(), 1, validity_bytes, slot.stream),
			                   "async set pipeline accumulator value validity");
		}
	}
	if (error) {
		return 1;
	}

	auto d_grids = slot.IsMapped() ? slot.mapped_grids.DeviceAs<int64_t>()
	                               : (slot.IsManaged() ? slot.managed_grids.As<int64_t>() : slot.grids.As<int64_t>());
	auto d_values = slot.IsMapped() ? slot.mapped_values.DeviceAs<double>()
	                                : (slot.IsManaged() ? slot.managed_values.As<double>() : slot.values.As<double>());
	auto d_value_validity = slot.IsMapped()
	                            ? slot.mapped_value_validity.DeviceAs<uint8_t>()
	                            : (slot.IsManaged() ? slot.managed_value_validity.As<uint8_t>()
	                                                : slot.value_validity.As<uint8_t>());
	auto d_grid_to_group = slot.IsMapped()
	                           ? slot.mapped_grid_to_group.DeviceAs<int32_t>()
	                           : (slot.IsManaged() ? slot.managed_grid_to_group.As<int32_t>()
	                                               : slot.grid_to_group.As<int32_t>());

	constexpr int THREADS_PER_BLOCK = 256;
	const auto blocks = static_cast<unsigned int>((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
	DuckDBGpuFusedLatAggKernel<<<blocks, THREADS_PER_BLOCK, 0, slot.stream>>>(
	    d_grids, d_values, d_value_validity, count, slot.grid_min, slot.grid_max, d_grid_to_group, slot.build_size,
	    slot.sums.As<double>(), slot.counts.As<unsigned long long>(), slot.row_counts.As<unsigned long long>());
	error |= CheckCuda(cudaGetLastError(), "launch pipeline accumulator fused lat aggregate kernel");
	if (error) {
		return 1;
	}

	slot.count += count;
	slot.active = true;
	return 0;
}

extern "C" int duckdb_gpu_fused_lat_agg_pipeline_prepare_input_i64_double(void *handle, uint32_t slot_idx,
                                                                          uint64_t capacity, int64_t **grids_out,
                                                                          double **values_out,
                                                                          uint8_t **value_validity_out) {
	if (!handle || !grids_out || !values_out || !value_validity_out || capacity == 0) {
		return 1;
	}

	auto state = reinterpret_cast<FusedLatAggPipelineState *>(handle);
	if (slot_idx >= state->slot_count) {
		return 1;
	}
	auto &slot = state->slots[slot_idx];
	if (slot.EnsureStream()) {
		return 1;
	}
	if (slot.IsDevice()) {
		return 1;
	}

	const auto grid_bytes = capacity * sizeof(int64_t);
	const auto value_bytes = capacity * sizeof(double);
	const auto validity_bytes = capacity * sizeof(uint8_t);

	int error = 0;
	if (slot.IsMapped()) {
		error |= slot.mapped_grids.Ensure(grid_bytes, "resize mapped direct pipeline grids");
		error |= slot.mapped_values.Ensure(value_bytes, "resize mapped direct pipeline values");
		error |= slot.mapped_value_validity.Ensure(validity_bytes, "resize mapped direct pipeline value validity");
		if (error) {
			return 1;
		}
		*grids_out = slot.mapped_grids.HostAs<int64_t>();
		*values_out = slot.mapped_values.HostAs<double>();
		*value_validity_out = slot.mapped_value_validity.HostAs<uint8_t>();
		return 0;
	}

	error |= slot.managed_grids.Ensure(grid_bytes, "resize managed direct pipeline grids");
	error |= slot.managed_values.Ensure(value_bytes, "resize managed direct pipeline values");
	error |= slot.managed_value_validity.Ensure(validity_bytes, "resize managed direct pipeline value validity");
	if (error) {
		return 1;
	}
	*grids_out = slot.managed_grids.As<int64_t>();
	*values_out = slot.managed_values.As<double>();
	*value_validity_out = slot.managed_value_validity.As<uint8_t>();
	return 0;
}

extern "C" int duckdb_gpu_fused_lat_agg_pipeline_submit_prepared_i64_double(void *handle, uint32_t slot_idx,
                                                                            uint64_t count) {
	if (!handle) {
		return 1;
	}
	if (count == 0) {
		return 0;
	}

	auto state = reinterpret_cast<FusedLatAggPipelineState *>(handle);
	if (slot_idx >= state->slot_count) {
		return 1;
	}
	auto &slot = state->slots[slot_idx];
	if (slot.EnsureStream()) {
		return 1;
	}
	if (slot.IsDevice() || slot.group_count == 0 || slot.build_size == 0) {
		return 1;
	}

	int64_t *d_grids = nullptr;
	double *d_values = nullptr;
	uint8_t *d_value_validity = nullptr;
	int32_t *d_grid_to_group = nullptr;
	if (slot.IsMapped()) {
		d_grids = slot.mapped_grids.DeviceAs<int64_t>();
		d_values = slot.mapped_values.DeviceAs<double>();
		d_value_validity = slot.mapped_value_validity.DeviceAs<uint8_t>();
		d_grid_to_group = slot.mapped_grid_to_group.DeviceAs<int32_t>();
	} else {
		d_grids = slot.managed_grids.As<int64_t>();
		d_values = slot.managed_values.As<double>();
		d_value_validity = slot.managed_value_validity.As<uint8_t>();
		d_grid_to_group = slot.managed_grid_to_group.As<int32_t>();
	}

	constexpr int THREADS_PER_BLOCK = 256;
	const auto blocks = static_cast<unsigned int>((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
	DuckDBGpuFusedLatAggKernel<<<blocks, THREADS_PER_BLOCK, 0, slot.stream>>>(
	    d_grids, d_values, d_value_validity, count, slot.grid_min, slot.grid_max, d_grid_to_group, slot.build_size,
	    slot.sums.As<double>(), slot.counts.As<unsigned long long>(), slot.row_counts.As<unsigned long long>());
	if (CheckCuda(cudaGetLastError(), "launch prepared pipeline fused lat aggregate kernel")) {
		return 1;
	}

	slot.count += count;
	slot.active = true;
	return 0;
}

extern "C" int duckdb_gpu_fused_lat_agg_multi_pipeline_prepare_input_i64_double(
    void *handle, uint32_t slot_idx, uint64_t capacity, uint64_t column_count, int64_t **grids_out,
    double **values_out, uint8_t **value_validity_out) {
	if (!handle || !grids_out || !values_out || !value_validity_out || capacity == 0 || column_count == 0) {
		return 1;
	}

	auto state = reinterpret_cast<FusedLatAggPipelineState *>(handle);
	if (slot_idx >= state->slot_count) {
		return 1;
	}
	auto &slot = state->slots[slot_idx];
	if (slot.EnsureStream()) {
		return 1;
	}
	if (slot.IsDevice()) {
		return 1;
	}

	const auto grid_bytes = capacity * sizeof(int64_t);
	const auto value_bytes = column_count * capacity * sizeof(double);
	const auto validity_bytes = column_count * capacity * sizeof(uint8_t);
	const bool assume_all_valid = AssumePayloadAllValid();

	int error = 0;
	if (slot.IsMapped()) {
		error |= slot.mapped_grids.Ensure(grid_bytes, "resize mapped direct multi pipeline grids");
		error |= slot.mapped_values.Ensure(value_bytes, "resize mapped direct multi pipeline values");
		if (!assume_all_valid) {
			error |= slot.mapped_value_validity.Ensure(validity_bytes, "resize mapped direct multi pipeline validity");
		}
		if (error) {
			return 1;
		}
		*grids_out = slot.mapped_grids.HostAs<int64_t>();
		*values_out = slot.mapped_values.HostAs<double>();
		*value_validity_out = assume_all_valid ? nullptr : slot.mapped_value_validity.HostAs<uint8_t>();
	} else {
		error |= slot.managed_grids.Ensure(grid_bytes, "resize managed direct multi pipeline grids");
		error |= slot.managed_values.Ensure(value_bytes, "resize managed direct multi pipeline values");
		if (!assume_all_valid) {
			error |= slot.managed_value_validity.Ensure(validity_bytes, "resize managed direct multi pipeline validity");
		}
		if (error) {
			return 1;
		}
		*grids_out = slot.managed_grids.As<int64_t>();
		*values_out = slot.managed_values.As<double>();
		*value_validity_out = assume_all_valid ? nullptr : slot.managed_value_validity.As<uint8_t>();
	}

	slot.column_count = column_count;
	slot.value_stride = capacity;
	return 0;
}

extern "C" int duckdb_gpu_fused_lat_agg_multi_pipeline_prepare_device_input_i64_double(
    void *handle, uint32_t slot_idx, uint64_t capacity, uint64_t column_count) {
	if (!handle || capacity == 0 || column_count == 0) {
		return 1;
	}

	auto state = reinterpret_cast<FusedLatAggPipelineState *>(handle);
	if (slot_idx >= state->slot_count) {
		return 1;
	}
	auto &slot = state->slots[slot_idx];
	if (slot.EnsureStream() || !slot.IsDevice()) {
		return 1;
	}

	const auto grid_bytes = capacity * sizeof(int64_t);
	const auto value_bytes = column_count * capacity * sizeof(double);
	const auto validity_bytes = column_count * capacity * sizeof(uint8_t);
	const bool assume_all_valid = AssumePayloadAllValid();

	int error = 0;
	error |= slot.grids.Ensure(grid_bytes, "resize device direct multi pipeline grids");
	error |= slot.values.Ensure(value_bytes, "resize device direct multi pipeline values");
	if (!assume_all_valid) {
		error |= slot.value_validity.Ensure(validity_bytes, "resize device direct multi pipeline validity");
	}
	if (error) {
		return 1;
	}

	slot.column_count = column_count;
	slot.value_stride = capacity;
	return 0;
}

extern "C" int duckdb_gpu_fused_lat_agg_multi_pipeline_copy_grids_i64(
    void *handle, uint32_t slot_idx, uint64_t dst_offset, const int64_t *grids, uint64_t count) {
	if (!handle || !grids || count == 0) {
		return 1;
	}

	auto state = reinterpret_cast<FusedLatAggPipelineState *>(handle);
	if (slot_idx >= state->slot_count) {
		return 1;
	}
	auto &slot = state->slots[slot_idx];
	if (slot.EnsureStream() || !slot.IsDevice() || slot.value_stride < dst_offset + count) {
		return 1;
	}

	auto dst = slot.grids.As<int64_t>() + dst_offset;
	return CheckCuda(cudaMemcpy(dst, grids, count * sizeof(int64_t), cudaMemcpyHostToDevice),
	                 "copy device direct multi pipeline grids");
}

extern "C" int duckdb_gpu_fused_lat_agg_multi_pipeline_copy_values_double(
    void *handle, uint32_t slot_idx, uint64_t column_idx, uint64_t dst_offset, const double *values,
    const uint8_t *value_validity, uint64_t count, int validity_all_valid) {
	if (!handle || !values || count == 0) {
		return 1;
	}

	auto state = reinterpret_cast<FusedLatAggPipelineState *>(handle);
	if (slot_idx >= state->slot_count) {
		return 1;
	}
	auto &slot = state->slots[slot_idx];
	if (slot.EnsureStream() || !slot.IsDevice() || column_idx >= slot.column_count ||
	    slot.value_stride < dst_offset + count) {
		return 1;
	}

	const auto offset = column_idx * slot.value_stride + dst_offset;
	auto dst_values = slot.values.As<double>() + offset;
	const bool assume_all_valid = AssumePayloadAllValid();
	int error = 0;
	error |= CheckCuda(cudaMemcpy(dst_values, values, count * sizeof(double), cudaMemcpyHostToDevice),
	                   "copy device direct multi pipeline values");
	if (!assume_all_valid && validity_all_valid) {
		auto dst_validity = slot.value_validity.As<uint8_t>() + offset;
		error |= CheckCuda(cudaMemset(dst_validity, 1, count * sizeof(uint8_t)),
		                   "set device direct multi pipeline validity");
	} else if (!assume_all_valid && value_validity) {
		auto dst_validity = slot.value_validity.As<uint8_t>() + offset;
		error |= CheckCuda(cudaMemcpy(dst_validity, value_validity, count * sizeof(uint8_t), cudaMemcpyHostToDevice),
		                   "copy device direct multi pipeline validity");
	} else if (!assume_all_valid) {
		return 1;
	}
	return error ? 1 : 0;
}

extern "C" int duckdb_gpu_fused_lat_agg_multi_pipeline_reset_i64_double(
    void *handle, uint32_t slot_idx, int64_t grid_min, int64_t grid_max, const int32_t *grid_to_group,
    uint64_t build_size, uint64_t group_count, uint64_t column_count) {
	if (!handle || !grid_to_group || group_count == 0 || column_count == 0) {
		return 1;
	}
	if (grid_max < grid_min || build_size == 0) {
		return 1;
	}

	auto state = reinterpret_cast<FusedLatAggPipelineState *>(handle);
	if (slot_idx >= state->slot_count) {
		return 1;
	}
	auto &slot = state->slots[slot_idx];
	if (slot.EnsureStream()) {
		return 1;
	}
	if (slot.active) {
		if (CheckCuda(cudaStreamSynchronize(slot.stream), "wait direct multi pipeline slot before accumulator reset")) {
			return 1;
		}
		slot.active = false;
	}

	const auto build_bytes = build_size * sizeof(int32_t);
	const auto output_count = column_count * group_count;
	const auto sum_bytes = output_count * sizeof(double);
	const auto output_count_bytes = output_count * sizeof(unsigned long long);
	const auto row_count_bytes = group_count * sizeof(unsigned long long);

	int error = 0;
	error |= slot.sums.Ensure(sum_bytes, "resize direct multi pipeline accumulator sums");
	error |= slot.counts.Ensure(output_count_bytes, "resize direct multi pipeline accumulator counts");
	error |= slot.row_counts.Ensure(row_count_bytes, "resize direct multi pipeline accumulator row counts");
	if (slot.IsMapped()) {
		error |= slot.mapped_grid_to_group.Ensure(build_bytes, "resize mapped direct multi grid to group");
	} else if (slot.IsManaged()) {
		error |= slot.managed_grid_to_group.Ensure(build_bytes, "resize managed direct multi grid to group");
	} else {
		error |= slot.grid_to_group.Ensure(build_bytes, "resize device direct multi grid to group");
	}
	if (error) {
		return 1;
	}

	if (slot.IsMapped()) {
		std::memcpy(slot.mapped_grid_to_group.HostAs<int32_t>(), grid_to_group, build_bytes);
	} else if (slot.IsManaged()) {
		std::memcpy(slot.managed_grid_to_group.As<int32_t>(), grid_to_group, build_bytes);
	} else {
		error |= CheckCuda(cudaMemcpyAsync(slot.grid_to_group.As<int32_t>(), grid_to_group, build_bytes,
		                                  cudaMemcpyHostToDevice, slot.stream),
		                   "async copy device direct multi grid to group");
	}
	error |= CheckCuda(cudaMemsetAsync(slot.sums.As<double>(), 0, sum_bytes, slot.stream),
	                   "async clear direct multi pipeline sums");
	error |= CheckCuda(cudaMemsetAsync(slot.counts.As<unsigned long long>(), 0, output_count_bytes, slot.stream),
	                   "async clear direct multi pipeline counts");
	error |= CheckCuda(cudaMemsetAsync(slot.row_counts.As<unsigned long long>(), 0, row_count_bytes, slot.stream),
	                   "async clear direct multi pipeline row counts");
	if (error) {
		return 1;
	}

	slot.count = 0;
	slot.group_count = group_count;
	slot.build_size = build_size;
	slot.column_count = column_count;
	slot.grid_min = grid_min;
	slot.grid_max = grid_max;
	slot.active = true;
	return 0;
}

extern "C" int duckdb_gpu_fused_lat_agg_multi_pipeline_submit_prepared_i64_double(
    void *handle, uint32_t slot_idx, uint64_t count, uint64_t column_count, uint64_t value_stride) {
	if (!handle || count == 0 || column_count == 0 || value_stride < count) {
		return 1;
	}

	auto state = reinterpret_cast<FusedLatAggPipelineState *>(handle);
	if (slot_idx >= state->slot_count) {
		return 1;
	}
	auto &slot = state->slots[slot_idx];
	if (slot.EnsureStream()) {
		return 1;
	}
	if (slot.group_count == 0 || slot.build_size == 0) {
		return 1;
	}

	int64_t *d_grids = nullptr;
	double *d_values = nullptr;
	uint8_t *d_value_validity = nullptr;
	int32_t *d_grid_to_group = nullptr;
	if (slot.IsMapped()) {
		d_grids = slot.mapped_grids.DeviceAs<int64_t>();
		d_values = slot.mapped_values.DeviceAs<double>();
		d_value_validity = AssumePayloadAllValid() ? nullptr : slot.mapped_value_validity.DeviceAs<uint8_t>();
		d_grid_to_group = slot.mapped_grid_to_group.DeviceAs<int32_t>();
	} else if (slot.IsManaged()) {
		d_grids = slot.managed_grids.As<int64_t>();
		d_values = slot.managed_values.As<double>();
		d_value_validity = AssumePayloadAllValid() ? nullptr : slot.managed_value_validity.As<uint8_t>();
		d_grid_to_group = slot.managed_grid_to_group.As<int32_t>();
	} else {
		d_grids = slot.grids.As<int64_t>();
		d_values = slot.values.As<double>();
		d_value_validity = AssumePayloadAllValid() ? nullptr : slot.value_validity.As<uint8_t>();
		d_grid_to_group = slot.grid_to_group.As<int32_t>();
	}

	constexpr int THREADS_PER_BLOCK = 256;
	const auto blocks = static_cast<unsigned int>((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
	DuckDBGpuFusedLatAggMultiStridedKernel<<<blocks, THREADS_PER_BLOCK, 0, slot.stream>>>(
	    d_grids, d_values, d_value_validity, column_count, value_stride, count, slot.grid_min, slot.grid_max,
	    d_grid_to_group, slot.build_size, slot.group_count, slot.sums.As<double>(),
	    slot.counts.As<unsigned long long>(), slot.row_counts.As<unsigned long long>());
	if (CheckCuda(cudaGetLastError(), "launch prepared direct multi pipeline fused lat aggregate kernel")) {
		return 1;
	}

	slot.count += count;
	slot.column_count = column_count;
	slot.value_stride = value_stride;
	slot.active = true;
	return 0;
}

extern "C" int duckdb_gpu_fused_lat_agg_multi_pipeline_wait(void *handle, uint32_t slot_idx, double *sum_out,
                                                            uint64_t *count_out, uint64_t *row_count_out) {
	if (!handle || !sum_out || !count_out || !row_count_out) {
		return 1;
	}

	auto state = reinterpret_cast<FusedLatAggPipelineState *>(handle);
	if (slot_idx >= state->slot_count) {
		return 1;
	}
	auto &slot = state->slots[slot_idx];
	if (!slot.active || slot.group_count == 0 || slot.column_count == 0) {
		return 1;
	}

	const auto output_count = slot.column_count * slot.group_count;
	const auto sum_bytes = output_count * sizeof(double);
	const auto output_count_bytes = output_count * sizeof(unsigned long long);
	const auto row_count_bytes = slot.group_count * sizeof(unsigned long long);
	int error = 0;
	error |= CheckCuda(cudaMemcpyAsync(sum_out, slot.sums.As<double>(), sum_bytes, cudaMemcpyDeviceToHost,
	                                  slot.stream),
	                   "async copy direct multi pipeline fused sums to host");
	error |= CheckCuda(cudaMemcpyAsync(count_out, slot.counts.As<unsigned long long>(), output_count_bytes,
	                                  cudaMemcpyDeviceToHost, slot.stream),
	                   "async copy direct multi pipeline fused counts to host");
	error |= CheckCuda(cudaMemcpyAsync(row_count_out, slot.row_counts.As<unsigned long long>(), row_count_bytes,
	                                  cudaMemcpyDeviceToHost, slot.stream),
	                   "async copy direct multi pipeline fused row counts to host");
	error |= CheckCuda(cudaStreamSynchronize(slot.stream), "sync direct multi pipeline fused slot");
	if (error) {
		return 1;
	}
	slot.active = false;
	return 0;
}

extern "C" int duckdb_gpu_fused_lat_agg_pipeline_sync_slot(void *handle, uint32_t slot_idx) {
	if (!handle) {
		return 1;
	}
	auto state = reinterpret_cast<FusedLatAggPipelineState *>(handle);
	if (slot_idx >= state->slot_count) {
		return 1;
	}
	auto &slot = state->slots[slot_idx];
	if (slot.EnsureStream()) {
		return 1;
	}
	return CheckCuda(cudaStreamSynchronize(slot.stream), "sync direct pipeline slot");
}

extern "C" int duckdb_gpu_fused_lat_agg_pipeline_wait(void *handle, uint32_t slot_idx, double *sum_out,
                                                       uint64_t *count_out, uint64_t *row_count_out) {
	if (!handle || !sum_out || !count_out || !row_count_out) {
		return 1;
	}

	auto state = reinterpret_cast<FusedLatAggPipelineState *>(handle);
	if (slot_idx >= state->slot_count) {
		return 1;
	}
	auto &slot = state->slots[slot_idx];
	if (!slot.active) {
		return 1;
	}

	const auto group_count = slot.group_count;
	const auto sum_bytes = group_count * sizeof(double);
	const auto count_bytes = group_count * sizeof(unsigned long long);
	int error = 0;
	error |= CheckCuda(cudaMemcpyAsync(sum_out, slot.sums.As<double>(), sum_bytes, cudaMemcpyDeviceToHost, slot.stream),
	                   "async copy pipeline fused sums to host");
	error |= CheckCuda(cudaMemcpyAsync(count_out, slot.counts.As<unsigned long long>(), count_bytes,
	                                  cudaMemcpyDeviceToHost, slot.stream),
	                   "async copy pipeline fused counts to host");
	error |= CheckCuda(cudaMemcpyAsync(row_count_out, slot.row_counts.As<unsigned long long>(), count_bytes,
	                                  cudaMemcpyDeviceToHost, slot.stream),
	                   "async copy pipeline fused row counts to host");
	error |= CheckCuda(cudaStreamSynchronize(slot.stream), "sync pipeline fused slot");
	if (error) {
		return 1;
	}
	slot.active = false;
	return 0;
}

extern "C" void duckdb_gpu_fused_lat_agg_pipeline_destroy(void *handle) {
	auto state = reinterpret_cast<FusedLatAggPipelineState *>(handle);
	delete state;
}
