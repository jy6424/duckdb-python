#include <cuda_runtime.h>

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

	int64_t *d_keys = nullptr;
	uint8_t *d_validity = nullptr;
	uint8_t *d_build_bitmap = nullptr;
	uint32_t *d_probe_sel = nullptr;
	uint32_t *d_build_sel = nullptr;
	unsigned long long *d_count = nullptr;

	const auto keys_bytes = count * sizeof(int64_t);
	const auto validity_bytes = count * sizeof(uint8_t);
	const auto bitmap_bytes = build_size * sizeof(uint8_t);
	const auto output_bytes = count * sizeof(uint32_t);

	int error = 0;
	error |= CheckCuda(cudaMalloc(&d_keys, keys_bytes), "malloc keys");
	error |= CheckCuda(cudaMalloc(&d_validity, validity_bytes), "malloc validity");
	error |= CheckCuda(cudaMalloc(&d_build_bitmap, bitmap_bytes), "malloc build bitmap");
	error |= CheckCuda(cudaMalloc(&d_probe_sel, output_bytes), "malloc probe selection");
	error |= CheckCuda(cudaMalloc(&d_build_sel, output_bytes), "malloc build selection");
	error |= CheckCuda(cudaMalloc(&d_count, sizeof(unsigned long long)), "malloc count");
	if (error) {
		goto cleanup;
	}

	error |= CheckCuda(cudaMemcpy(d_keys, keys, keys_bytes, cudaMemcpyHostToDevice), "copy keys to device");
	error |=
	    CheckCuda(cudaMemcpy(d_validity, validity, validity_bytes, cudaMemcpyHostToDevice), "copy validity to device");
	error |= CheckCuda(cudaMemcpy(d_build_bitmap, build_bitmap, bitmap_bytes, cudaMemcpyHostToDevice),
	                   "copy build bitmap to device");
	error |= CheckCuda(cudaMemset(d_count, 0, sizeof(unsigned long long)), "clear count");
	if (error) {
		goto cleanup;
	}

	{
		constexpr int THREADS_PER_BLOCK = 256;
		const auto blocks = static_cast<unsigned int>((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
		DuckDBGpuProbeI64Kernel<<<blocks, THREADS_PER_BLOCK>>>(d_keys, d_validity, count, min_value, max_value,
		                                                       d_build_bitmap, build_size, d_probe_sel, d_build_sel,
		                                                       d_count);
		error |= CheckCuda(cudaGetLastError(), "launch probe kernel");
		error |= CheckCuda(cudaDeviceSynchronize(), "synchronize probe kernel");
	}
	if (error) {
		goto cleanup;
	}

	unsigned long long result_count = 0;
	error |= CheckCuda(cudaMemcpy(&result_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost),
	                   "copy count to host");
	if (error || result_count > count) {
		error = 1;
		goto cleanup;
	}

	error |= CheckCuda(cudaMemcpy(probe_sel_out, d_probe_sel, result_count * sizeof(uint32_t), cudaMemcpyDeviceToHost),
	                   "copy probe selection to host");
	error |= CheckCuda(cudaMemcpy(build_sel_out, d_build_sel, result_count * sizeof(uint32_t), cudaMemcpyDeviceToHost),
	                   "copy build selection to host");
	*out_count = static_cast<uint64_t>(result_count);

cleanup:
	cudaFree(d_keys);
	cudaFree(d_validity);
	cudaFree(d_build_bitmap);
	cudaFree(d_probe_sel);
	cudaFree(d_build_sel);
	cudaFree(d_count);
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

	uint64_t *d_addresses = nullptr;
	uint8_t *d_validity = nullptr;
	uint64_t *d_unique_addresses = nullptr;
	uint64_t *d_counts = nullptr;
	unsigned long long *d_unique_count = nullptr;

	const auto addresses_bytes = count * sizeof(uint64_t);
	const auto validity_bytes = count * sizeof(uint8_t);
	const auto output_bytes = count * sizeof(uint64_t);

	int error = 0;
	error |= CheckCuda(cudaMalloc(&d_addresses, addresses_bytes), "malloc groupby count addresses");
	error |= CheckCuda(cudaMalloc(&d_validity, validity_bytes), "malloc groupby count validity");
	error |= CheckCuda(cudaMalloc(&d_unique_addresses, output_bytes), "malloc groupby count unique addresses");
	error |= CheckCuda(cudaMalloc(&d_counts, output_bytes), "malloc groupby count output counts");
	error |= CheckCuda(cudaMalloc(&d_unique_count, sizeof(unsigned long long)), "malloc groupby count unique count");
	if (error) {
		goto cleanup;
	}

	error |= CheckCuda(cudaMemcpy(d_addresses, addresses, addresses_bytes, cudaMemcpyHostToDevice),
	                   "copy groupby count addresses to device");
	error |= CheckCuda(cudaMemcpy(d_validity, validity, validity_bytes, cudaMemcpyHostToDevice),
	                   "copy groupby count validity to device");
	error |= CheckCuda(cudaMemset(d_unique_count, 0, sizeof(unsigned long long)), "clear groupby count unique count");
	if (error) {
		goto cleanup;
	}

	{
		constexpr int THREADS_PER_BLOCK = 256;
		const auto blocks = static_cast<unsigned int>((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
		DuckDBGpuGroupByCountKernel<<<blocks, THREADS_PER_BLOCK>>>(d_addresses, d_validity, count, d_unique_addresses,
		                                                           d_counts, d_unique_count);
		error |= CheckCuda(cudaGetLastError(), "launch groupby count kernel");
		error |= CheckCuda(cudaDeviceSynchronize(), "synchronize groupby count kernel");
	}
	if (error) {
		goto cleanup;
	}

	unsigned long long result_count = 0;
	error |= CheckCuda(cudaMemcpy(&result_count, d_unique_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost),
	                   "copy groupby count unique count to host");
	if (error || result_count > count) {
		error = 1;
		goto cleanup;
	}

	error |= CheckCuda(cudaMemcpy(unique_addresses_out, d_unique_addresses, result_count * sizeof(uint64_t),
	                              cudaMemcpyDeviceToHost),
	                   "copy groupby count unique addresses to host");
	error |= CheckCuda(cudaMemcpy(counts_out, d_counts, result_count * sizeof(uint64_t), cudaMemcpyDeviceToHost),
	                   "copy groupby count counts to host");
	*unique_count_out = static_cast<uint64_t>(result_count);

cleanup:
	cudaFree(d_addresses);
	cudaFree(d_validity);
	cudaFree(d_unique_addresses);
	cudaFree(d_counts);
	cudaFree(d_unique_count);
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

	uint16_t *d_keys = nullptr;
	uint8_t *d_validity = nullptr;
	uint8_t *d_build_bitmap = nullptr;
	uint32_t *d_probe_sel = nullptr;
	uint32_t *d_build_sel = nullptr;
	unsigned long long *d_count = nullptr;

	const auto keys_bytes = count * sizeof(uint16_t);
	const auto validity_bytes = count * sizeof(uint8_t);
	const auto bitmap_bytes = build_size * sizeof(uint8_t);
	const auto output_bytes = count * sizeof(uint32_t);

	int error = 0;
	error |= CheckCuda(cudaMalloc(&d_keys, keys_bytes), "malloc u16 keys");
	error |= CheckCuda(cudaMalloc(&d_validity, validity_bytes), "malloc u16 validity");
	error |= CheckCuda(cudaMalloc(&d_build_bitmap, bitmap_bytes), "malloc u16 build bitmap");
	error |= CheckCuda(cudaMalloc(&d_probe_sel, output_bytes), "malloc u16 probe selection");
	error |= CheckCuda(cudaMalloc(&d_build_sel, output_bytes), "malloc u16 build selection");
	error |= CheckCuda(cudaMalloc(&d_count, sizeof(unsigned long long)), "malloc u16 count");
	if (error) {
		goto cleanup;
	}

	error |= CheckCuda(cudaMemcpy(d_keys, keys, keys_bytes, cudaMemcpyHostToDevice), "copy u16 keys to device");
	error |= CheckCuda(cudaMemcpy(d_validity, validity, validity_bytes, cudaMemcpyHostToDevice),
	                   "copy u16 validity to device");
	error |= CheckCuda(cudaMemcpy(d_build_bitmap, build_bitmap, bitmap_bytes, cudaMemcpyHostToDevice),
	                   "copy u16 build bitmap to device");
	error |= CheckCuda(cudaMemset(d_count, 0, sizeof(unsigned long long)), "clear u16 count");
	if (error) {
		goto cleanup;
	}

	{
		constexpr int THREADS_PER_BLOCK = 256;
		const auto blocks = static_cast<unsigned int>((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
		DuckDBGpuProbeU16Kernel<<<blocks, THREADS_PER_BLOCK>>>(d_keys, d_validity, count, min_value, max_value,
		                                                       d_build_bitmap, build_size, d_probe_sel, d_build_sel,
		                                                       d_count);
		error |= CheckCuda(cudaGetLastError(), "launch u16 probe kernel");
		error |= CheckCuda(cudaDeviceSynchronize(), "synchronize u16 probe kernel");
	}
	if (error) {
		goto cleanup;
	}

	unsigned long long result_count = 0;
	error |= CheckCuda(cudaMemcpy(&result_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost),
	                   "copy u16 count to host");
	if (error || result_count > count) {
		error = 1;
		goto cleanup;
	}

	error |= CheckCuda(cudaMemcpy(probe_sel_out, d_probe_sel, result_count * sizeof(uint32_t), cudaMemcpyDeviceToHost),
	                   "copy u16 probe selection to host");
	error |= CheckCuda(cudaMemcpy(build_sel_out, d_build_sel, result_count * sizeof(uint32_t), cudaMemcpyDeviceToHost),
	                   "copy u16 build selection to host");
	*out_count = static_cast<uint64_t>(result_count);

cleanup:
	cudaFree(d_keys);
	cudaFree(d_validity);
	cudaFree(d_build_bitmap);
	cudaFree(d_probe_sel);
	cudaFree(d_build_sel);
	cudaFree(d_count);
	return error ? 1 : 0;
}
