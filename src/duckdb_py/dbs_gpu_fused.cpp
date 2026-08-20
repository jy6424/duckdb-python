#include "duckdb_python/pybind11/pybind_wrapper.hpp"

#include "duckdb.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/vector.hpp"
#include "../../external/duckdb/extension/parquet/include/parquet_reader.hpp"

#include <chrono>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <dlfcn.h>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <utility>

#if defined(__linux__)
#include <fcntl.h>
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#endif

namespace py = pybind11;

namespace duckdb {

namespace {

using FusedLatAggFunc = int (*)(const int64_t *grids, const double *values, const uint8_t *value_validity,
                                uint64_t count, int64_t grid_min, int64_t grid_max, const int32_t *grid_to_group,
                                uint64_t build_size, uint64_t group_count, double *sum_out, uint64_t *count_out,
                                uint64_t *row_count_out);
using FusedLatAggMultiFunc = int (*)(const int64_t *grids, const double *values, const uint8_t *value_validity,
                                     uint64_t column_count, uint64_t count, int64_t grid_min, int64_t grid_max,
                                     const int32_t *grid_to_group, uint64_t build_size, uint64_t group_count,
                                     double *sum_out, uint64_t *count_out, uint64_t *row_count_out);
using FusedLatAggMultiStridedFunc = int (*)(const int64_t *grids, const double *values,
                                            const uint8_t *value_validity, uint64_t column_count,
                                            uint64_t value_stride, uint64_t count, int64_t grid_min,
                                            int64_t grid_max, const int32_t *grid_to_group, uint64_t build_size,
                                            uint64_t group_count, double *sum_out, uint64_t *count_out,
                                            uint64_t *row_count_out);
using FusedLatAggPipelineCreateFunc = void *(*)(uint32_t slot_count, int mapped);
using FusedLatAggPipelineSubmitFunc = int (*)(void *handle, uint32_t slot_idx, const int64_t *grids,
                                              const double *values, const uint8_t *value_validity, uint64_t count,
                                              int64_t grid_min, int64_t grid_max, const int32_t *grid_to_group,
                                              uint64_t build_size, uint64_t group_count);
using FusedLatAggPipelineResetFunc = int (*)(void *handle, uint32_t slot_idx, int64_t grid_min, int64_t grid_max,
                                             const int32_t *grid_to_group, uint64_t build_size,
                                             uint64_t group_count);
using FusedLatAggPipelineSubmitAccumulateFunc = int (*)(void *handle, uint32_t slot_idx, const int64_t *grids,
                                                        const double *values, const uint8_t *value_validity,
                                                        uint64_t count);
using FusedLatAggPipelinePrepareInputFunc = int (*)(void *handle, uint32_t slot_idx, uint64_t capacity,
                                                    int64_t **grids_out, double **values_out,
                                                    uint8_t **value_validity_out);
using FusedLatAggPipelineSubmitPreparedFunc = int (*)(void *handle, uint32_t slot_idx, uint64_t count);
using FusedLatAggPipelineSyncSlotFunc = int (*)(void *handle, uint32_t slot_idx);
using FusedLatAggPipelineWaitFunc = int (*)(void *handle, uint32_t slot_idx, double *sum_out, uint64_t *count_out,
                                            uint64_t *row_count_out);
using FusedLatAggPipelineDestroyFunc = void (*)(void *handle);
using FusedLatAggMultiPipelinePrepareInputFunc = int (*)(void *handle, uint32_t slot_idx, uint64_t capacity,
                                                         uint64_t column_count, int64_t **grids_out,
                                                         double **values_out, uint8_t **value_validity_out);
using FusedLatAggMultiPipelinePrepareDeviceInputFunc = int (*)(void *handle, uint32_t slot_idx, uint64_t capacity,
                                                               uint64_t column_count);
using FusedLatAggMultiPipelineCopyGridsFunc = int (*)(void *handle, uint32_t slot_idx, uint64_t dst_offset,
                                                      const int64_t *grids, uint64_t count);
using FusedLatAggMultiPipelineCopyValuesFunc = int (*)(void *handle, uint32_t slot_idx, uint64_t column_idx,
                                                       uint64_t dst_offset, const double *values,
                                                       const uint8_t *value_validity, uint64_t count,
                                                       int validity_all_valid);
using FusedLatAggMultiPipelineResetFunc = int (*)(void *handle, uint32_t slot_idx, int64_t grid_min,
                                                  int64_t grid_max, const int32_t *grid_to_group,
                                                  uint64_t build_size, uint64_t group_count,
                                                  uint64_t column_count);
using FusedLatAggMultiPipelineSubmitPreparedFunc = int (*)(void *handle, uint32_t slot_idx, uint64_t count,
                                                           uint64_t column_count, uint64_t value_stride);
using FusedLatAggMultiPipelineWaitFunc = int (*)(void *handle, uint32_t slot_idx, double *sum_out,
                                                 uint64_t *count_out, uint64_t *row_count_out);

struct FusedLatAggPipelineFuncs {
	FusedLatAggPipelineCreateFunc create = nullptr;
	FusedLatAggPipelineSubmitFunc submit = nullptr;
	FusedLatAggPipelineResetFunc reset = nullptr;
	FusedLatAggPipelineSubmitAccumulateFunc submit_accumulate = nullptr;
	FusedLatAggPipelinePrepareInputFunc prepare_input = nullptr;
	FusedLatAggPipelineSubmitPreparedFunc submit_prepared = nullptr;
	FusedLatAggPipelineSyncSlotFunc sync_slot = nullptr;
	FusedLatAggPipelineWaitFunc wait = nullptr;
	FusedLatAggPipelineDestroyFunc destroy = nullptr;
};

struct FusedLatAggMultiDirectPipelineFuncs {
	FusedLatAggPipelineCreateFunc create = nullptr;
	FusedLatAggMultiPipelinePrepareInputFunc prepare_input = nullptr;
	FusedLatAggMultiPipelinePrepareDeviceInputFunc prepare_device_input = nullptr;
	FusedLatAggMultiPipelineCopyGridsFunc copy_grids = nullptr;
	FusedLatAggMultiPipelineCopyValuesFunc copy_values = nullptr;
	FusedLatAggMultiPipelineResetFunc reset = nullptr;
	FusedLatAggMultiPipelineSubmitPreparedFunc submit_prepared = nullptr;
	FusedLatAggPipelineSyncSlotFunc sync_slot = nullptr;
	FusedLatAggMultiPipelineWaitFunc wait = nullptr;
	FusedLatAggPipelineDestroyFunc destroy = nullptr;
};

static string EscapeSQLString(const string &value) {
	string result;
	result.reserve(value.size());
	for (auto c : value) {
		if (c == '\'') {
			result += "''";
		} else {
			result += c;
		}
	}
	return result;
}

static string QuoteIdentifier(const string &value) {
	string result = "\"";
	for (auto c : value) {
		if (c == '"') {
			result += "\"\"";
		} else {
			result += c;
		}
	}
	result += "\"";
	return result;
}

static string ParentPath(const string &path) {
	auto slash = path.find_last_of('/');
	if (slash == string::npos) {
		return ".";
	}
	if (slash == 0) {
		return "/";
	}
	return path.substr(0, slash);
}

static vector<string> SplitFactPathGroup(const string &fact_path) {
	vector<string> paths;
	idx_t start = 0;
	while (start <= fact_path.size()) {
		auto next = fact_path.find('\n', start);
		auto end = next == string::npos ? fact_path.size() : next;
		if (end > start) {
			paths.push_back(fact_path.substr(start, end - start));
		}
		if (next == string::npos) {
			break;
		}
		start = next + 1;
	}
	if (paths.empty()) {
		paths.push_back(fact_path);
	}
	return paths;
}

static string FirstFactPath(const string &fact_path) {
	auto paths = SplitFactPathGroup(fact_path);
	return paths[0];
}

static string BuildReadParquetExpression(const string &fact_path) {
	auto paths = SplitFactPathGroup(fact_path);
	const string options =
	    ", union_by_name=false, hive_partitioning=false, filename=false, file_row_number=false, binary_as_string=false";
	if (paths.size() == 1) {
		return "read_parquet('" + EscapeSQLString(paths[0]) + "'" + options + ")";
	}

	string result = "read_parquet([";
	for (idx_t path = 0; path < paths.size(); path++) {
		if (path > 0) {
			result += ", ";
		}
		result += "'" + EscapeSQLString(paths[path]) + "'";
	}
	result += "]" + options + ")";
	return result;
}

static string BuildReadParquetExpressionForPath(const string &path) {
	return BuildReadParquetExpression(path);
}

static string ResolveDimensionPath(const string &fact_path, const string &dimension_file) {
	if (!dimension_file.empty() && (dimension_file[0] == '/' || dimension_file.find('/') != string::npos)) {
		return dimension_file;
	}
	return ParentPath(FirstFactPath(fact_path)) + "/" + dimension_file;
}

static idx_t ReadEnvIdx(const char *name, idx_t default_value) {
	auto value = std::getenv(name);
	if (!value || !value[0]) {
		return default_value;
	}
	char *end = nullptr;
	auto parsed = std::strtoull(value, &end, 10);
	if (!end || *end != '\0' || parsed == 0) {
		return default_value;
	}
	return static_cast<idx_t>(parsed);
}

static bool ReadEnvFlag(const char *name, bool default_value = false) {
	auto value = std::getenv(name);
	if (!value || !value[0]) {
		return default_value;
	}
	return std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 || std::strcmp(value, "TRUE") == 0 ||
	       std::strcmp(value, "yes") == 0 || std::strcmp(value, "YES") == 0 || std::strcmp(value, "on") == 0 ||
	       std::strcmp(value, "ON") == 0;
}

static string ReadEnvString(const char *name, const string &default_value) {
	auto value = std::getenv(name);
	if (!value || !value[0]) {
		return default_value;
	}
	return value;
}

static bool UseRawGpuFetch() {
	return ReadEnvFlag("DUCKDB_GPU_FETCH_RAW", false);
}

static bool InferGridFromRowOrder() {
	return ReadEnvFlag("DUCKDB_GPU_INFER_GRID_FROM_ROW_ORDER", false);
}

static unique_ptr<DataChunk> FetchGpuPipelineChunk(QueryResult &result) {
	if (UseRawGpuFetch()) {
		return result.FetchRaw();
	}
	return result.Fetch();
}

template <class T>
struct UnifiedColumnReader {
	UnifiedColumnReader(Vector &vector, idx_t count) {
		vector.ToUnifiedFormat(count, format);
		data = UnifiedVectorFormat::GetData<T>(format);
	}

	idx_t Index(idx_t row) const {
		return format.sel->get_index(row);
	}

	bool RowIsValid(idx_t row) const {
		return format.validity.RowIsValid(Index(row));
	}

	T Value(idx_t row) const {
		return data[Index(row)];
	}

	UnifiedVectorFormat format;
	const T *data = nullptr;
};

template <class T>
static void CopyUnifiedValues(T *target, const UnifiedColumnReader<T> &source, idx_t source_offset, idx_t count) {
	for (idx_t row = 0; row < count; row++) {
		target[row] = source.Value(source_offset + row);
	}
}

template <class T>
static void CopyUnifiedValidityToBytes(uint8_t *target, const UnifiedColumnReader<T> &source, idx_t source_offset,
                                       idx_t count) {
	for (idx_t row = 0; row < count; row++) {
		target[row] = source.RowIsValid(source_offset + row) ? 1 : 0;
	}
}

static FusedLatAggFunc LoadFusedLatAgg(const string &path_p, bool mapped) {
	static void *device_handle = nullptr;
	static void *mapped_handle = nullptr;
	static FusedLatAggFunc device_func = nullptr;
	static FusedLatAggFunc mapped_func = nullptr;

	void **handle_ptr = mapped ? &mapped_handle : &device_handle;
	auto &func = mapped ? mapped_func : device_func;
	if (func) {
		return func;
	}

	string path = path_p;
	if (path.empty()) {
		const char *env_path = std::getenv("DUCKDB_GPU_PROBE_LIB");
		if (env_path && env_path[0]) {
			path = env_path;
		} else {
			path = "libduckdb_gpu_probe.so";
		}
	}

	*handle_ptr = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
	if (!*handle_ptr) {
		throw InvalidInputException("Failed to load GPU helper library '%s': %s", path, dlerror());
	}

	const char *symbol = mapped ? "duckdb_gpu_fused_lat_agg_i64_double_mapped" : "duckdb_gpu_fused_lat_agg_i64_double";
	func = reinterpret_cast<FusedLatAggFunc>(dlsym(*handle_ptr, symbol));
	if (!func) {
		throw InvalidInputException("Failed to load GPU helper symbol '%s': %s", symbol, dlerror());
	}
	return func;
}

static FusedLatAggMultiFunc LoadFusedLatAggMulti(const string &path_p, bool mapped) {
	static void *device_handle = nullptr;
	static void *mapped_handle = nullptr;
	static FusedLatAggMultiFunc device_func = nullptr;
	static FusedLatAggMultiFunc mapped_func = nullptr;

	void **handle_ptr = mapped ? &mapped_handle : &device_handle;
	auto &func = mapped ? mapped_func : device_func;
	if (func) {
		return func;
	}

	string path = path_p;
	if (path.empty()) {
		const char *env_path = std::getenv("DUCKDB_GPU_PROBE_LIB");
		if (env_path && env_path[0]) {
			path = env_path;
		} else {
			path = "libduckdb_gpu_probe.so";
		}
	}

	*handle_ptr = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
	if (!*handle_ptr) {
		throw InvalidInputException("Failed to load GPU helper library '%s': %s", path, dlerror());
	}

	const char *symbol =
	    mapped ? "duckdb_gpu_fused_lat_agg_multi_i64_double_mapped" : "duckdb_gpu_fused_lat_agg_multi_i64_double";
	func = reinterpret_cast<FusedLatAggMultiFunc>(dlsym(*handle_ptr, symbol));
	if (!func) {
		throw InvalidInputException("Failed to load GPU helper symbol '%s': %s", symbol, dlerror());
	}
	return func;
}

static FusedLatAggMultiStridedFunc LoadFusedLatAggMultiStrided(const string &path_p, bool mapped) {
	static void *device_handle = nullptr;
	static void *mapped_handle = nullptr;
	static FusedLatAggMultiStridedFunc device_func = nullptr;
	static FusedLatAggMultiStridedFunc mapped_func = nullptr;

	void **handle_ptr = mapped ? &mapped_handle : &device_handle;
	auto &func = mapped ? mapped_func : device_func;
	if (func) {
		return func;
	}

	string path = path_p;
	if (path.empty()) {
		const char *env_path = std::getenv("DUCKDB_GPU_PROBE_LIB");
		if (env_path && env_path[0]) {
			path = env_path;
		} else {
			path = "libduckdb_gpu_probe.so";
		}
	}

	*handle_ptr = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
	if (!*handle_ptr) {
		throw InvalidInputException("Failed to load GPU helper library '%s': %s", path, dlerror());
	}

	const char *symbol = mapped ? "duckdb_gpu_fused_lat_agg_multi_i64_double_strided_mapped" :
	                             "duckdb_gpu_fused_lat_agg_multi_i64_double_strided";
	func = reinterpret_cast<FusedLatAggMultiStridedFunc>(dlsym(*handle_ptr, symbol));
	if (!func) {
		throw InvalidInputException("Failed to load GPU helper symbol '%s': %s", symbol, dlerror());
	}
	return func;
}

static FusedLatAggPipelineFuncs LoadFusedLatAggPipeline(const string &path_p) {
	static void *handle = nullptr;
	static FusedLatAggPipelineFuncs funcs;
	if (funcs.create && funcs.submit && funcs.reset && funcs.submit_accumulate && funcs.prepare_input &&
	    funcs.submit_prepared && funcs.sync_slot && funcs.wait && funcs.destroy) {
		return funcs;
	}

	string path = path_p;
	if (path.empty()) {
		const char *env_path = std::getenv("DUCKDB_GPU_PROBE_LIB");
		if (env_path && env_path[0]) {
			path = env_path;
		} else {
			path = "libduckdb_gpu_probe.so";
		}
	}

	handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
	if (!handle) {
		throw InvalidInputException("Failed to load GPU helper library '%s': %s", path, dlerror());
	}

	funcs.create = reinterpret_cast<FusedLatAggPipelineCreateFunc>(
	    dlsym(handle, "duckdb_gpu_fused_lat_agg_pipeline_create"));
	funcs.submit = reinterpret_cast<FusedLatAggPipelineSubmitFunc>(
	    dlsym(handle, "duckdb_gpu_fused_lat_agg_pipeline_submit_i64_double"));
	funcs.reset = reinterpret_cast<FusedLatAggPipelineResetFunc>(
	    dlsym(handle, "duckdb_gpu_fused_lat_agg_pipeline_reset_i64_double"));
	funcs.submit_accumulate = reinterpret_cast<FusedLatAggPipelineSubmitAccumulateFunc>(
	    dlsym(handle, "duckdb_gpu_fused_lat_agg_pipeline_submit_accumulate_i64_double"));
	funcs.prepare_input = reinterpret_cast<FusedLatAggPipelinePrepareInputFunc>(
	    dlsym(handle, "duckdb_gpu_fused_lat_agg_pipeline_prepare_input_i64_double"));
	funcs.submit_prepared = reinterpret_cast<FusedLatAggPipelineSubmitPreparedFunc>(
	    dlsym(handle, "duckdb_gpu_fused_lat_agg_pipeline_submit_prepared_i64_double"));
	funcs.sync_slot =
	    reinterpret_cast<FusedLatAggPipelineSyncSlotFunc>(dlsym(handle, "duckdb_gpu_fused_lat_agg_pipeline_sync_slot"));
	funcs.wait =
	    reinterpret_cast<FusedLatAggPipelineWaitFunc>(dlsym(handle, "duckdb_gpu_fused_lat_agg_pipeline_wait"));
	funcs.destroy =
	    reinterpret_cast<FusedLatAggPipelineDestroyFunc>(dlsym(handle, "duckdb_gpu_fused_lat_agg_pipeline_destroy"));
	if (!funcs.create || !funcs.submit || !funcs.reset || !funcs.submit_accumulate || !funcs.prepare_input ||
	    !funcs.submit_prepared || !funcs.sync_slot || !funcs.wait || !funcs.destroy) {
		throw InvalidInputException("Failed to load GPU fused pipeline symbols from '%s'", path);
	}
	return funcs;
}

static FusedLatAggMultiDirectPipelineFuncs LoadFusedLatAggMultiDirectPipeline(const string &path_p,
                                                                             bool require_device_direct = false) {
	static void *handle = nullptr;
	static FusedLatAggMultiDirectPipelineFuncs funcs;
	if (funcs.create && funcs.prepare_input && funcs.reset && funcs.submit_prepared && funcs.sync_slot &&
	    funcs.wait && funcs.destroy &&
	    (!require_device_direct || (funcs.prepare_device_input && funcs.copy_grids && funcs.copy_values))) {
		return funcs;
	}

	string path = path_p;
	if (path.empty()) {
		const char *env_path = std::getenv("DUCKDB_GPU_PROBE_LIB");
		if (env_path && env_path[0]) {
			path = env_path;
		} else {
			path = "libduckdb_gpu_probe.so";
		}
	}

	handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
	if (!handle) {
		throw InvalidInputException("Failed to load GPU helper library '%s': %s", path, dlerror());
	}

	funcs.create = reinterpret_cast<FusedLatAggPipelineCreateFunc>(
	    dlsym(handle, "duckdb_gpu_fused_lat_agg_pipeline_create"));
	funcs.prepare_input = reinterpret_cast<FusedLatAggMultiPipelinePrepareInputFunc>(
	    dlsym(handle, "duckdb_gpu_fused_lat_agg_multi_pipeline_prepare_input_i64_double"));
	funcs.prepare_device_input = reinterpret_cast<FusedLatAggMultiPipelinePrepareDeviceInputFunc>(
	    dlsym(handle, "duckdb_gpu_fused_lat_agg_multi_pipeline_prepare_device_input_i64_double"));
	funcs.copy_grids = reinterpret_cast<FusedLatAggMultiPipelineCopyGridsFunc>(
	    dlsym(handle, "duckdb_gpu_fused_lat_agg_multi_pipeline_copy_grids_i64"));
	funcs.copy_values = reinterpret_cast<FusedLatAggMultiPipelineCopyValuesFunc>(
	    dlsym(handle, "duckdb_gpu_fused_lat_agg_multi_pipeline_copy_values_double"));
	funcs.reset = reinterpret_cast<FusedLatAggMultiPipelineResetFunc>(
	    dlsym(handle, "duckdb_gpu_fused_lat_agg_multi_pipeline_reset_i64_double"));
	funcs.submit_prepared = reinterpret_cast<FusedLatAggMultiPipelineSubmitPreparedFunc>(
	    dlsym(handle, "duckdb_gpu_fused_lat_agg_multi_pipeline_submit_prepared_i64_double"));
	funcs.sync_slot =
	    reinterpret_cast<FusedLatAggPipelineSyncSlotFunc>(dlsym(handle, "duckdb_gpu_fused_lat_agg_pipeline_sync_slot"));
	funcs.wait = reinterpret_cast<FusedLatAggMultiPipelineWaitFunc>(
	    dlsym(handle, "duckdb_gpu_fused_lat_agg_multi_pipeline_wait"));
	funcs.destroy =
	    reinterpret_cast<FusedLatAggPipelineDestroyFunc>(dlsym(handle, "duckdb_gpu_fused_lat_agg_pipeline_destroy"));
	if (!funcs.create || !funcs.prepare_input || !funcs.reset || !funcs.submit_prepared || !funcs.sync_slot ||
	    !funcs.wait || !funcs.destroy) {
		throw InvalidInputException("Failed to load GPU fused multi direct pipeline symbols from '%s'", path);
	}
	if (require_device_direct && (!funcs.prepare_device_input || !funcs.copy_grids || !funcs.copy_values)) {
		throw InvalidInputException("Failed to load GPU fused multi device-direct pipeline symbols from '%s'", path);
	}
	return funcs;
}

static unique_ptr<QueryResult> RunStreamingQuery(Connection &connection, const string &query) {
	auto result = connection.SendQuery(query, QueryResultOutputType::ALLOW_STREAMING);
	if (result->HasError()) {
		result->ThrowError();
	}
	return result;
}

static void ConfigurePipelineReaderConnection(Connection &connection) {
	auto preserve_result = connection.Query("SET preserve_insertion_order=false");
	if (preserve_result->HasError()) {
		preserve_result->ThrowError();
	}
	auto value = std::getenv("DUCKDB_GPU_READER_DUCKDB_THREADS");
	if (value && value[0]) {
		char *end = nullptr;
		auto parsed = std::strtoull(value, &end, 10);
		if (!end || *end != '\0' || parsed == 0) {
			throw InvalidInputException("DUCKDB_GPU_READER_DUCKDB_THREADS must be a positive integer");
		}
		auto result = connection.Query("SET threads=" + std::to_string(parsed));
		if (result->HasError()) {
			result->ThrowError();
		}
	}
	if (ReadEnvFlag("DUCKDB_PARQUET_ASYNC_PREFETCH", false) ||
	    ReadEnvFlag("DUCKDB_PARQUET_PREFETCH_ALL_FILES", false)) {
		auto result = connection.Query("SET prefetch_all_parquet_files=true");
		if (result->HasError()) {
			result->ThrowError();
		}
	}
}

#if defined(__linux__) && defined(SYS_io_uring_setup) && defined(SYS_io_uring_enter)
class IoUringPrefetcher {
public:
	~IoUringPrefetcher() {
		Close();
	}

	bool Init(uint32_t entries) {
		params = {};
		ring_fd = static_cast<int>(syscall(SYS_io_uring_setup, entries, &params));
		if (ring_fd < 0) {
			return false;
		}

		sq_ring_size = params.sq_off.array + params.sq_entries * sizeof(uint32_t);
		cq_ring_size = params.cq_off.cqes + params.cq_entries * sizeof(struct io_uring_cqe);
		sqes_size = params.sq_entries * sizeof(struct io_uring_sqe);

		sq_ring = mmap(nullptr, sq_ring_size, PROT_READ | PROT_WRITE, MAP_SHARED, ring_fd, IORING_OFF_SQ_RING);
		cq_ring = mmap(nullptr, cq_ring_size, PROT_READ | PROT_WRITE, MAP_SHARED, ring_fd, IORING_OFF_CQ_RING);
		sqes = static_cast<struct io_uring_sqe *>(
		    mmap(nullptr, sqes_size, PROT_READ | PROT_WRITE, MAP_SHARED, ring_fd, IORING_OFF_SQES));
		if (sq_ring == MAP_FAILED || cq_ring == MAP_FAILED || reinterpret_cast<void *>(sqes) == MAP_FAILED) {
			Close();
			return false;
		}

		sq_head = reinterpret_cast<uint32_t *>(static_cast<char *>(sq_ring) + params.sq_off.head);
		sq_tail = reinterpret_cast<uint32_t *>(static_cast<char *>(sq_ring) + params.sq_off.tail);
		sq_ring_mask = reinterpret_cast<uint32_t *>(static_cast<char *>(sq_ring) + params.sq_off.ring_mask);
		sq_array = reinterpret_cast<uint32_t *>(static_cast<char *>(sq_ring) + params.sq_off.array);

		cq_head = reinterpret_cast<uint32_t *>(static_cast<char *>(cq_ring) + params.cq_off.head);
		cq_tail = reinterpret_cast<uint32_t *>(static_cast<char *>(cq_ring) + params.cq_off.tail);
		cq_ring_mask = reinterpret_cast<uint32_t *>(static_cast<char *>(cq_ring) + params.cq_off.ring_mask);
		cqes = reinterpret_cast<struct io_uring_cqe *>(static_cast<char *>(cq_ring) + params.cq_off.cqes);
		iovecs.resize(params.sq_entries);
		return true;
	}

	void Close() {
		if (sq_ring && sq_ring != MAP_FAILED) {
			munmap(sq_ring, sq_ring_size);
		}
		if (cq_ring && cq_ring != MAP_FAILED) {
			munmap(cq_ring, cq_ring_size);
		}
		if (sqes && reinterpret_cast<void *>(sqes) != MAP_FAILED) {
			munmap(sqes, sqes_size);
		}
		if (ring_fd >= 0) {
			close(ring_fd);
		}
		sq_ring = nullptr;
		cq_ring = nullptr;
		sqes = nullptr;
		ring_fd = -1;
	}

	bool SubmitRead(int fd, uint64_t offset, void *buffer, uint32_t length, uint64_t user_data) {
		auto tail = *sq_tail;
		auto index = tail & *sq_ring_mask;
		auto &sqe = sqes[index];
		std::memset(&sqe, 0, sizeof(sqe));
#if defined(IORING_OP_READ)
		sqe.opcode = IORING_OP_READ;
		sqe.addr = reinterpret_cast<uint64_t>(buffer);
		sqe.len = length;
#else
		iovecs[index].iov_base = buffer;
		iovecs[index].iov_len = length;
		sqe.opcode = IORING_OP_READV;
		sqe.addr = reinterpret_cast<uint64_t>(&iovecs[index]);
		sqe.len = 1;
#endif
		sqe.fd = fd;
		sqe.off = offset;
		sqe.user_data = user_data;
		sq_array[index] = index;
		std::atomic_thread_fence(std::memory_order_release);
		*sq_tail = tail + 1;
		pending_submissions++;
		return true;
	}

	bool SubmitAndWait(uint32_t wait_nr) {
		auto submit_count = pending_submissions;
		pending_submissions = 0;
		auto result = syscall(SYS_io_uring_enter, ring_fd, submit_count, wait_nr, IORING_ENTER_GETEVENTS, nullptr, 0);
		return result >= 0;
	}

	uint32_t DrainCompletions() {
		uint32_t completed = 0;
		auto head = *cq_head;
		std::atomic_thread_fence(std::memory_order_acquire);
		auto tail = *cq_tail;
		while (head != tail) {
			auto &cqe = cqes[head & *cq_ring_mask];
			(void)cqe;
			head++;
			completed++;
		}
		*cq_head = head;
		return completed;
	}

private:
	int ring_fd = -1;
	struct io_uring_params params;
	void *sq_ring = nullptr;
	void *cq_ring = nullptr;
	struct io_uring_sqe *sqes = nullptr;
	struct io_uring_cqe *cqes = nullptr;
	size_t sq_ring_size = 0;
	size_t cq_ring_size = 0;
	size_t sqes_size = 0;
	uint32_t *sq_head = nullptr;
	uint32_t *sq_tail = nullptr;
	uint32_t *sq_ring_mask = nullptr;
	uint32_t *sq_array = nullptr;
	uint32_t *cq_head = nullptr;
	uint32_t *cq_tail = nullptr;
	uint32_t *cq_ring_mask = nullptr;
	uint32_t pending_submissions = 0;
	vector<struct iovec> iovecs;
};

static void PrefetchFileWithIoUring(const string &path) {
	auto queue_depth = static_cast<uint32_t>(ReadEnvIdx("DUCKDB_GPU_IO_URING_QD", 8));
	queue_depth = std::max<uint32_t>(1, std::min<uint32_t>(queue_depth, 64));
	auto block_bytes = static_cast<uint32_t>(ReadEnvIdx("DUCKDB_GPU_IO_URING_BLOCK_BYTES", 1024 * 1024));
	block_bytes = std::max<uint32_t>(4096, block_bytes);

	auto fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
	if (fd < 0) {
		return;
	}

	struct stat file_stat;
	if (fstat(fd, &file_stat) != 0 || !S_ISREG(file_stat.st_mode) || file_stat.st_size <= 0) {
		close(fd);
		return;
	}

	IoUringPrefetcher ring;
	if (!ring.Init(queue_depth)) {
		close(fd);
		return;
	}

	vector<vector<char>> buffers;
	buffers.reserve(queue_depth);
	for (uint32_t idx = 0; idx < queue_depth; idx++) {
		buffers.emplace_back(block_bytes);
	}

	uint64_t offset = 0;
	auto file_size = static_cast<uint64_t>(file_stat.st_size);
	uint64_t request_id = 0;

	while (offset < file_size) {
		uint32_t submitted = 0;
		while (offset < file_size && submitted < queue_depth) {
			auto buffer_idx = submitted;
			auto remaining = file_size - offset;
			auto request = static_cast<uint32_t>(std::min<uint64_t>(remaining, block_bytes));
			if (!ring.SubmitRead(fd, offset, buffers[buffer_idx].data(), request, request_id)) {
				break;
			}
			offset += request;
			request_id++;
			submitted++;
		}
		if (submitted == 0 || !ring.SubmitAndWait(submitted)) {
			break;
		}
		uint32_t completed = 0;
		while (completed < submitted) {
			completed += ring.DrainCompletions();
			if (completed < submitted && !ring.SubmitAndWait(1)) {
				break;
			}
		}
	}
	close(fd);
}
#endif

static void PrefetchFileBestEffort(const string &path, const string &method) {
#if defined(__linux__)
	if (method == "io_uring") {
#if defined(SYS_io_uring_setup) && defined(SYS_io_uring_enter)
		PrefetchFileWithIoUring(path);
#endif
		return;
	}

	auto fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
	if (fd < 0) {
		return;
	}

	struct stat file_stat;
	if (fstat(fd, &file_stat) != 0 || !S_ISREG(file_stat.st_mode)) {
		close(fd);
		return;
	}

	const bool use_fadvise = method == "fadvise" || method == "both";
	const bool use_readahead = method == "readahead" || method == "both";
	if (use_fadvise) {
		(void)posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED);
	}
	if (use_readahead && file_stat.st_size > 0) {
#if defined(SYS_readahead)
		constexpr uint64_t READAHEAD_CHUNK_SIZE = 64ULL * 1024ULL * 1024ULL;
		uint64_t offset = 0;
		uint64_t remaining = static_cast<uint64_t>(file_stat.st_size);
		while (remaining > 0) {
			auto request = static_cast<size_t>(std::min<uint64_t>(remaining, READAHEAD_CHUNK_SIZE));
			(void)syscall(SYS_readahead, fd, static_cast<off_t>(offset), request);
			offset += request;
			remaining -= request;
		}
#endif
	}
	close(fd);
#else
	(void)path;
	(void)method;
#endif
}

static void PrefetchPipelineFiles(vector<string> fact_paths, string dimension_file) {
	auto method = ReadEnvString("DUCKDB_GPU_PREFETCH_METHOD", "both");
	if (method != "fadvise" && method != "readahead" && method != "both" && method != "io_uring") {
		method = "both";
	}
	std::set<string> prefetch_paths;
	const bool prefetch_dimension_files = ReadEnvFlag("DUCKDB_GPU_PREFETCH_DIMENSION_FILES", true);
	const bool reuse_dimension_mapping = ReadEnvFlag("DUCKDB_GPU_REUSE_DIMENSION_MAPPING", false);
	string reused_dimension_path;
	for (auto &fact_path : fact_paths) {
		auto paths = SplitFactPathGroup(fact_path);
		for (auto &path : paths) {
			prefetch_paths.insert(path);
		}
		if (prefetch_dimension_files) {
			auto dimension_path = ResolveDimensionPath(fact_path, dimension_file);
			if (!reuse_dimension_mapping) {
				prefetch_paths.insert(dimension_path);
			} else if (reused_dimension_path.empty()) {
				reused_dimension_path = dimension_path;
				prefetch_paths.insert(reused_dimension_path);
			}
		}
	}
	for (auto &path : prefetch_paths) {
		PrefetchFileBestEffort(path, method);
	}
}

class ScopedThread {
public:
	~ScopedThread() {
		Join();
	}

	template <class... ARGS>
	void Start(ARGS &&...args) {
		thread = std::thread(std::forward<ARGS>(args)...);
	}

	void Join() {
		if (thread.joinable()) {
			thread.join();
		}
	}

private:
	std::thread thread;
};

struct GroupMapping {
	int64_t join_min = 0;
	int64_t join_max = 0;
	vector<int32_t> join_to_group;
	vector<double> group_values;
};

static GroupMapping ReadGroupMapping(Connection &connection, const string &dimension_path, const string &join_key,
                                     const string &group_column) {
	auto query = StringUtil::Format(
	    "SELECT %s AS join_key, %s AS group_value FROM %s",
	    QuoteIdentifier(join_key), QuoteIdentifier(group_column), BuildReadParquetExpressionForPath(dimension_path));
	auto result = RunStreamingQuery(connection, query);

	vector<int64_t> join_keys;
	vector<double> group_values_per_row;
	bool first = true;
	int64_t join_min = 0;
	int64_t join_max = 0;

	while (true) {
		auto chunk = FetchGpuPipelineChunk(*result);
		if (!chunk || chunk->size() == 0) {
			break;
		}
		auto count = chunk->size();
		UnifiedColumnReader<int64_t> join_data(chunk->data[0], count);
		UnifiedColumnReader<double> group_data(chunk->data[1], count);
		for (idx_t row = 0; row < count; row++) {
			if (!join_data.RowIsValid(row) || !group_data.RowIsValid(row)) {
				continue;
			}
			auto join_value = join_data.Value(row);
			auto group_value = group_data.Value(row);
			join_keys.push_back(join_value);
			group_values_per_row.push_back(group_value);
			if (first) {
				join_min = join_value;
				join_max = join_value;
				first = false;
			} else {
				join_min = std::min(join_min, join_value);
				join_max = std::max(join_max, join_value);
			}
		}
	}
	if (join_keys.empty() || join_max < join_min) {
		throw InvalidInputException("Invalid or empty dimension parquet file '%s'", dimension_path);
	}

	std::map<double, int32_t> group_ids;
	for (auto group_value : group_values_per_row) {
		if (group_ids.find(group_value) == group_ids.end()) {
			auto group_id = static_cast<int32_t>(group_ids.size());
			group_ids[group_value] = group_id;
		}
	}

	auto build_size = static_cast<idx_t>(join_max - join_min + 1);
	GroupMapping mapping;
	mapping.join_min = join_min;
	mapping.join_max = join_max;
	mapping.join_to_group.assign(build_size, -1);
	mapping.group_values.reserve(group_ids.size());
	for (auto &entry : group_ids) {
		mapping.group_values.push_back(entry.first);
	}
	for (idx_t row = 0; row < join_keys.size(); row++) {
		auto offset = static_cast<idx_t>(join_keys[row] - join_min);
		mapping.join_to_group[offset] = group_ids[group_values_per_row[row]];
	}
	return mapping;
}

static GroupMapping ReadGroupMappingByRowNumber(Connection &connection, const string &dimension_path,
                                                const string &group_column) {
	auto query = StringUtil::Format("SELECT %s AS group_value FROM %s", QuoteIdentifier(group_column),
	                                BuildReadParquetExpressionForPath(dimension_path));
	auto result = RunStreamingQuery(connection, query);

	GroupMapping mapping;
	std::map<double, int32_t> group_ids;
	while (true) {
		auto chunk = FetchGpuPipelineChunk(*result);
		if (!chunk || chunk->size() == 0) {
			break;
		}
		auto count = chunk->size();
		UnifiedColumnReader<double> group_data(chunk->data[0], count);
		mapping.join_to_group.reserve(mapping.join_to_group.size() + count);
		for (idx_t row = 0; row < count; row++) {
			if (!group_data.RowIsValid(row)) {
				throw InvalidInputException("row-order grid inference requires non-null dimension groups");
			}
			auto group_value = group_data.Value(row);
			auto entry = group_ids.find(group_value);
			if (entry == group_ids.end()) {
				auto group_id = static_cast<int32_t>(group_ids.size());
				entry = group_ids.emplace(group_value, group_id).first;
			}
			mapping.join_to_group.push_back(entry->second);
		}
	}
	if (mapping.join_to_group.empty()) {
		throw InvalidInputException("Invalid or empty dimension parquet file '%s'", dimension_path);
	}
	mapping.join_min = 0;
	mapping.join_max = static_cast<int64_t>(mapping.join_to_group.size() - 1);
	mapping.group_values.reserve(group_ids.size());
	for (auto &entry : group_ids) {
		mapping.group_values.push_back(entry.first);
	}
	return mapping;
}

static GroupMapping ReadGroupMappingForCurrentMode(Connection &connection, const string &dimension_path,
                                                   const string &join_key, const string &group_column) {
	if (InferGridFromRowOrder()) {
		return ReadGroupMappingByRowNumber(connection, dimension_path, group_column);
	}
	return ReadGroupMapping(connection, dimension_path, join_key, group_column);
}

static string DimensionMappingCacheKey(const string &dimension_path, const string &dimension_file,
                                       const string &join_key, const string &group_column,
                                       bool reuse_dimension_mapping) {
	string prefix = InferGridFromRowOrder() ? "row_order:" : "grid_key:";
	if (reuse_dimension_mapping) {
		return prefix + "reuse:" + dimension_file + ":" + join_key + ":" + group_column;
	}
	return prefix + "path:" + dimension_path + ":" + join_key + ":" + group_column;
}

struct ProbeColumns {
	vector<int64_t> join_keys;
	vector<double> values;
	vector<uint8_t> validity;
};

struct MultiProbeColumns {
	vector<int64_t> join_keys;
	vector<vector<double>> values;
	vector<vector<uint8_t>> validity;
};

template <class T>
class BlockingQueue {
public:
	explicit BlockingQueue(idx_t capacity_p = 0) : capacity(capacity_p) {
	}

	void Push(T item) {
		{
			std::unique_lock<std::mutex> guard(lock);
			condition.wait(guard, [&] { return closed || capacity == 0 || items.size() < capacity; });
			if (closed) {
				return;
			}
			items.push_back(std::move(item));
		}
		condition.notify_all();
	}

	bool Pop(T &result) {
		std::unique_lock<std::mutex> guard(lock);
		condition.wait(guard, [&] { return closed || !items.empty(); });
		if (items.empty()) {
			return false;
		}
		result = std::move(items.front());
		items.pop_front();
		condition.notify_all();
		return true;
	}

	void Close() {
		{
			std::lock_guard<std::mutex> guard(lock);
			closed = true;
		}
		condition.notify_all();
	}

private:
	std::mutex lock;
	std::condition_variable condition;
	std::deque<T> items;
	idx_t capacity;
	bool closed = false;
};

struct PipelineRawBatch {
	string fact_path;
	std::shared_ptr<GroupMapping> mapping;
	vector<int64_t> join_keys;
	vector<double> values;
	vector<uint8_t> value_validity;
	vector<uint8_t> join_validity;
};

struct PipelineInputBatch {
	string fact_path;
	std::shared_ptr<GroupMapping> mapping;
	ProbeColumns probe;
};

struct DirectPipelineInputBatch {
	string fact_path;
	std::shared_ptr<GroupMapping> mapping;
	idx_t slot = 0;
	idx_t count = 0;
};

struct DirectPipelineBuffer {
	bool active = false;
	string fact_path;
	std::shared_ptr<GroupMapping> mapping;
	idx_t slot = 0;
	idx_t count = 0;
	idx_t chunks = 0;
	int64_t *join_keys = nullptr;
	double *values = nullptr;
	uint8_t *validity = nullptr;
};

struct PipelineOutputBatch {
	string fact_path;
	std::shared_ptr<GroupMapping> mapping;
	vector<double> sums;
	vector<uint64_t> counts;
	vector<uint64_t> row_counts;
};

struct PipelinePendingBatch {
	bool active = false;
	idx_t slot = 0;
	string fact_path;
	std::shared_ptr<GroupMapping> mapping;
	ProbeColumns probe;
	vector<double> sums;
	vector<uint64_t> counts;
	vector<uint64_t> row_counts;
};

struct MultiPipelineRawBatch {
	string fact_path;
	std::shared_ptr<GroupMapping> mapping;
	vector<int64_t> join_keys;
	vector<vector<double>> values;
	vector<vector<uint8_t>> validity;
	vector<uint8_t> join_validity;
};

struct MultiPipelineChunkBatch {
	string fact_path;
	std::shared_ptr<GroupMapping> mapping;
	std::unique_ptr<DataChunk> chunk;
	idx_t row_base = 0;
};

struct MultiPipelineInputBatch {
	string fact_path;
	std::shared_ptr<GroupMapping> mapping;
	vector<int64_t> join_keys;
	vector<double> values;
	vector<uint8_t> validity;
	idx_t row_count = 0;
	idx_t value_stride = 0;
};

struct DirectMultiPipelineInputBatch {
	string fact_path;
	std::shared_ptr<GroupMapping> mapping;
	idx_t slot = 0;
	idx_t row_count = 0;
	idx_t value_stride = 0;
	idx_t column_count = 0;
};

struct DirectMultiPipelineBuffer {
	bool active = false;
	bool device_direct = false;
	string fact_path;
	std::shared_ptr<GroupMapping> mapping;
	idx_t slot = 0;
	idx_t row_count = 0;
	idx_t chunks = 0;
	idx_t value_stride = 0;
	idx_t column_count = 0;
	int64_t *join_keys = nullptr;
	double *values = nullptr;
	uint8_t *validity = nullptr;
};

struct MultiPipelineOutputBatch {
	string fact_path;
	std::shared_ptr<GroupMapping> mapping;
	vector<double> sums;
	vector<uint64_t> counts;
	vector<uint64_t> row_counts;
	idx_t column_count = 0;
};

struct MultiPipelineAccumulatedResult {
	idx_t column_count = 0;
	uint64_t total_rows = 0;
	vector<double> group_values;
	vector<uint64_t> row_counts;
	vector<double> sums;
	vector<uint64_t> counts;
	std::map<double, idx_t> group_index;
};

struct MultiPipelineStageTimers {
	std::mutex lock;
	double read_time = 0;
	double prepare_time = 0;
	double gpu_time = 0;
	double merge_time = 0;
	double read_setup_time = 0;
	double read_connection_time = 0;
	double read_mapping_lock_time = 0;
	double read_mapping_time = 0;
	double read_query_build_time = 0;
	double read_query_submit_time = 0;
	double read_fetch_time = 0;
	double read_push_time = 0;
	double read_thread_max_time = 0;
	double prepare_pop_time = 0;
	double prepare_work_time = 0;
	double prepare_push_time = 0;
	double gpu_pop_time = 0;
	double gpu_work_time = 0;
	double gpu_push_time = 0;
	double merge_pop_time = 0;
	double merge_work_time = 0;
	uint64_t read_chunks = 0;
	uint64_t prepared_batches = 0;
	uint64_t gpu_batches = 0;
	uint64_t merged_batches = 0;
	uint64_t dimension_mapping_reads = 0;
	uint64_t dimension_mapping_reuses = 0;
};

static double ElapsedSeconds(std::chrono::steady_clock::time_point start) {
	return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

static string BuildProbeQuery(const string &fact_path, const string &join_key, const string &payload_column) {
	return StringUtil::Format(
	    "SELECT %s::BIGINT AS join_key, COALESCE(%s, 0)::DOUBLE AS value, "
	    "CASE WHEN %s IS NULL THEN 0 ELSE 1 END::UTINYINT AS value_valid FROM %s",
	    QuoteIdentifier(join_key), QuoteIdentifier(payload_column), QuoteIdentifier(payload_column),
	    BuildReadParquetExpression(fact_path));
}

static void AppendProbeChunk(DataChunk &chunk, ProbeColumns &columns) {
	auto count = chunk.size();
	UnifiedColumnReader<int64_t> join_data(chunk.data[0], count);
	UnifiedColumnReader<double> value_data(chunk.data[1], count);
	UnifiedColumnReader<uint8_t> valid_data(chunk.data[2], count);

	columns.join_keys.clear();
	columns.values.clear();
	columns.validity.clear();
	columns.join_keys.reserve(count);
	columns.values.reserve(count);
	columns.validity.reserve(count);

	for (idx_t row = 0; row < count; row++) {
		if (!join_data.RowIsValid(row)) {
			continue;
		}
		columns.join_keys.push_back(join_data.Value(row));
		columns.values.push_back(value_data.Value(row));
		columns.validity.push_back(valid_data.Value(row));
	}
}

static void CopyValidityToBytes(uint8_t *target, const ValidityMask &validity, idx_t source_offset, idx_t count) {
	if (validity.AllValid()) {
		std::memset(target, 1, count * sizeof(uint8_t));
		return;
	}
	for (idx_t row = 0; row < count; row++) {
		target[row] = validity.RowIsValid(source_offset + row) ? 1 : 0;
	}
}

static idx_t MultiPayloadColumnOffset() {
	return InferGridFromRowOrder() ? 0 : 1;
}

static int64_t RowOrderJoinKey(const GroupMapping &mapping, idx_t row_number) {
	auto mapping_size = mapping.join_to_group.size();
	if (mapping_size == 0) {
		throw InvalidInputException("row-order grid inference requires a non-empty dimension mapping");
	}
	return static_cast<int64_t>(row_number % mapping_size);
}

static void AppendMultiRawProbeChunk(DataChunk &chunk, MultiPipelineRawBatch &batch, idx_t column_count) {
	auto count = chunk.size();
	UnifiedColumnReader<int64_t> join_data(chunk.data[0], count);

	batch.join_keys.resize(count);
	batch.join_validity.resize(count);
	batch.values.resize(column_count);
	batch.validity.resize(column_count);
	for (idx_t column = 0; column < column_count; column++) {
		batch.values[column].resize(count);
		batch.validity[column].resize(count);
		UnifiedColumnReader<double> value_data(chunk.data[1 + column], count);
		for (idx_t row = 0; row < count; row++) {
			batch.values[column][row] = value_data.Value(row);
		}
		CopyUnifiedValidityToBytes(batch.validity[column].data(), value_data, 0, count);
	}

	for (idx_t row = 0; row < count; row++) {
		batch.join_keys[row] = join_data.Value(row);
		batch.join_validity[row] = join_data.RowIsValid(row) ? 1 : 0;
	}
}

static void AppendRawProbeChunk(DataChunk &chunk, PipelineRawBatch &batch) {
	auto count = chunk.size();
	UnifiedColumnReader<int64_t> join_data(chunk.data[0], count);
	UnifiedColumnReader<double> value_data(chunk.data[1], count);
	UnifiedColumnReader<uint8_t> valid_data(chunk.data[2], count);

	batch.join_keys.resize(count);
	batch.values.resize(count);
	batch.value_validity.resize(count);
	batch.join_validity.resize(count);

	for (idx_t row = 0; row < count; row++) {
		batch.join_keys[row] = join_data.Value(row);
		batch.values[row] = value_data.Value(row);
		batch.value_validity[row] = valid_data.Value(row);
		batch.join_validity[row] = join_data.RowIsValid(row) ? 1 : 0;
	}
}

static idx_t PreparedRowCount(const PipelineInputBatch &batch) {
	return batch.probe.join_keys.size();
}

static idx_t PreparedRowCount(const MultiPipelineInputBatch &batch) {
	return batch.row_count;
}

static void AppendPreparedRows(PipelineInputBatch &batch, PipelineRawBatch raw) {
	if (batch.fact_path.empty()) {
		batch.fact_path = std::move(raw.fact_path);
		batch.mapping = std::move(raw.mapping);
	}
	auto count = raw.join_keys.size();
	batch.probe.join_keys.reserve(batch.probe.join_keys.size() + count);
	batch.probe.values.reserve(batch.probe.values.size() + count);
	batch.probe.validity.reserve(batch.probe.validity.size() + count);

	for (idx_t row = 0; row < count; row++) {
		if (raw.join_validity[row] == 0) {
			continue;
		}
		batch.probe.join_keys.push_back(raw.join_keys[row]);
		batch.probe.values.push_back(raw.values[row]);
		batch.probe.validity.push_back(raw.value_validity[row]);
	}
}

static void AppendMultiPreparedRows(MultiPipelineInputBatch &batch, MultiPipelineRawBatch raw, idx_t column_count,
                                    idx_t target_batch_rows) {
	if (batch.fact_path.empty()) {
		batch.fact_path = std::move(raw.fact_path);
		batch.mapping = std::move(raw.mapping);
		batch.value_stride = target_batch_rows;
		batch.join_keys.reserve(target_batch_rows);
		batch.values.resize(column_count * target_batch_rows);
		batch.validity.resize(column_count * target_batch_rows);
	}
	auto count = raw.join_keys.size();

	for (idx_t row = 0; row < count; row++) {
		if (raw.join_validity[row] == 0) {
			continue;
		}
		auto output_row = batch.row_count;
		if (output_row >= target_batch_rows) {
			throw InvalidInputException("GPU fused multi pipeline batch row capacity exceeded");
		}
		batch.join_keys.push_back(raw.join_keys[row]);
		for (idx_t column = 0; column < column_count; column++) {
			auto offset = column * batch.value_stride + output_row;
			batch.values[offset] = raw.values[column][row];
			batch.validity[offset] = raw.validity[column][row];
		}
		batch.row_count++;
	}
}

static void StartMultiPipelineInputBatch(MultiPipelineInputBatch &batch, const string &fact_path,
                                         std::shared_ptr<GroupMapping> mapping, idx_t column_count,
                                         idx_t target_batch_rows) {
	batch = MultiPipelineInputBatch();
	batch.fact_path = fact_path;
	batch.mapping = std::move(mapping);
	batch.value_stride = target_batch_rows;
	batch.join_keys.reserve(target_batch_rows);
	batch.values.resize(column_count * target_batch_rows);
	batch.validity.resize(column_count * target_batch_rows);
}

static void AppendMultiPreparedChunkRows(MultiPipelineInputBatch &batch, DataChunk &chunk, idx_t chunk_row_base,
                                         idx_t &row_offset, idx_t column_count, idx_t target_batch_rows) {
	auto count = chunk.size();
	auto payload_offset = MultiPayloadColumnOffset();
	if (!UseRawGpuFetch()) {
		const int64_t *join_data = nullptr;
		ValidityMask *join_validity = nullptr;
		if (!InferGridFromRowOrder()) {
			join_data = FlatVector::GetData<int64_t>(chunk.data[0]);
			join_validity = &FlatVector::Validity(chunk.data[0]);
		}

		vector<const double *> value_data;
		vector<ValidityMask *> value_validity;
		value_data.reserve(column_count);
		value_validity.reserve(column_count);
		for (idx_t column = 0; column < column_count; column++) {
			value_data.push_back(FlatVector::GetData<double>(chunk.data[payload_offset + column]));
			value_validity.push_back(&FlatVector::Validity(chunk.data[payload_offset + column]));
		}

		if (InferGridFromRowOrder() || join_validity->AllValid()) {
			auto remaining_rows = count - row_offset;
			auto remaining_capacity = target_batch_rows - batch.row_count;
			auto append_count = std::min(remaining_rows, remaining_capacity);
			if (append_count == 0) {
				return;
			}

			auto output_row = batch.row_count;
			if (InferGridFromRowOrder()) {
				for (idx_t row = 0; row < append_count; row++) {
					batch.join_keys.push_back(RowOrderJoinKey(*batch.mapping, chunk_row_base + row_offset + row));
				}
			} else {
				batch.join_keys.insert(batch.join_keys.end(), join_data + row_offset,
				                       join_data + row_offset + append_count);
			}
			for (idx_t column = 0; column < column_count; column++) {
				auto output_offset = column * batch.value_stride + output_row;
				std::memcpy(batch.values.data() + output_offset, value_data[column] + row_offset,
				            append_count * sizeof(double));
				CopyValidityToBytes(batch.validity.data() + output_offset, *value_validity[column], row_offset,
				                    append_count);
			}
			batch.row_count += append_count;
			row_offset += append_count;
			return;
		}

		while (row_offset < count && batch.row_count < target_batch_rows) {
			auto row = row_offset++;
			if (!join_validity->RowIsValid(row)) {
				continue;
			}

			auto output_row = batch.row_count;
			batch.join_keys.push_back(join_data[row]);
			for (idx_t column = 0; column < column_count; column++) {
				auto offset = column * batch.value_stride + output_row;
				batch.values[offset] = value_data[column][row];
				batch.validity[offset] = value_validity[column]->RowIsValid(row) ? 1 : 0;
			}
			batch.row_count++;
		}
		return;
	}

	std::unique_ptr<UnifiedColumnReader<int64_t>> join_data;
	if (!InferGridFromRowOrder()) {
		join_data = make_uniq<UnifiedColumnReader<int64_t>>(chunk.data[0], count);
	}

	vector<UnifiedColumnReader<double>> value_data;
	value_data.reserve(column_count);
	for (idx_t column = 0; column < column_count; column++) {
		value_data.emplace_back(chunk.data[payload_offset + column], count);
	}

	if (InferGridFromRowOrder() || join_data->format.validity.AllValid()) {
		auto remaining_rows = count - row_offset;
		auto remaining_capacity = target_batch_rows - batch.row_count;
		auto append_count = std::min(remaining_rows, remaining_capacity);
		if (append_count == 0) {
			return;
		}

		auto output_row = batch.row_count;
		for (idx_t row = 0; row < append_count; row++) {
			if (InferGridFromRowOrder()) {
				batch.join_keys.push_back(RowOrderJoinKey(*batch.mapping, chunk_row_base + row_offset + row));
			} else {
				batch.join_keys.push_back(join_data->Value(row_offset + row));
			}
		}
		for (idx_t column = 0; column < column_count; column++) {
			auto output_offset = column * batch.value_stride + output_row;
			CopyUnifiedValues(batch.values.data() + output_offset, value_data[column], row_offset, append_count);
			CopyUnifiedValidityToBytes(batch.validity.data() + output_offset, value_data[column], row_offset,
			                           append_count);
		}
		batch.row_count += append_count;
		row_offset += append_count;
		return;
	}

	while (row_offset < count && batch.row_count < target_batch_rows) {
		auto row = row_offset++;
		if (!join_data->RowIsValid(row)) {
			continue;
		}

		auto output_row = batch.row_count;
		batch.join_keys.push_back(join_data->Value(row));
		for (idx_t column = 0; column < column_count; column++) {
			auto offset = column * batch.value_stride + output_row;
			batch.values[offset] = value_data[column].Value(row);
			batch.validity[offset] = value_data[column].RowIsValid(row) ? 1 : 0;
		}
		batch.row_count++;
	}
}

static void StartDirectMultiPipelineBuffer(FusedLatAggMultiDirectPipelineFuncs pipeline, void *handle,
                                           BlockingQueue<idx_t> &free_slots, idx_t target_batch_rows,
                                           idx_t column_count, const MultiPipelineChunkBatch &chunk_batch,
                                           DirectMultiPipelineBuffer &current, bool device_direct) {
	idx_t slot = 0;
	if (!free_slots.Pop(slot)) {
		throw InvalidInputException("GPU fused direct multi pipeline slot queue closed");
	}

	int64_t *join_keys = nullptr;
	double *values = nullptr;
	uint8_t *validity = nullptr;
	if (device_direct) {
		auto rc = pipeline.prepare_device_input(handle, static_cast<uint32_t>(slot),
		                                        static_cast<uint64_t>(target_batch_rows),
		                                        static_cast<uint64_t>(column_count));
		if (rc != 0) {
			throw InvalidInputException("GPU fused device-direct multi pipeline input preparation failed for '%s'",
			                            chunk_batch.fact_path);
		}
	} else {
		auto rc = pipeline.prepare_input(handle, static_cast<uint32_t>(slot), static_cast<uint64_t>(target_batch_rows),
		                                 static_cast<uint64_t>(column_count), &join_keys, &values, &validity);
		if (rc != 0 || !join_keys || !values || !validity) {
			throw InvalidInputException("GPU fused direct multi pipeline input preparation failed for '%s'",
			                            chunk_batch.fact_path);
		}
	}

	current = DirectMultiPipelineBuffer();
	current.active = true;
	current.device_direct = device_direct;
	current.fact_path = chunk_batch.fact_path;
	current.mapping = chunk_batch.mapping;
	current.slot = slot;
	current.value_stride = target_batch_rows;
	current.column_count = column_count;
	current.join_keys = join_keys;
	current.values = values;
	current.validity = validity;
}

static void AppendDeviceDirectMultiPreparedChunkRows(FusedLatAggMultiDirectPipelineFuncs pipeline, void *handle,
                                                     DirectMultiPipelineBuffer &batch, DataChunk &chunk,
                                                     idx_t chunk_row_base, idx_t &row_offset, idx_t column_count,
                                                     idx_t target_batch_rows, vector<uint8_t> &validity_scratch) {
	auto count = chunk.size();
	auto payload_offset = MultiPayloadColumnOffset();
	if (!UseRawGpuFetch()) {
		const int64_t *join_data = nullptr;
		if (!InferGridFromRowOrder()) {
			join_data = FlatVector::GetData<int64_t>(chunk.data[0]);
			auto &join_validity = FlatVector::Validity(chunk.data[0]);
			if (!join_validity.AllValid()) {
				throw InvalidInputException("pipeline-device-direct requires non-null join keys");
			}
		}
		if (!InferGridFromRowOrder() && !join_data) {
			throw InvalidInputException("pipeline-device-direct requires non-null join keys");
		}

		auto remaining_rows = count - row_offset;
		auto remaining_capacity = target_batch_rows - batch.row_count;
		auto append_count = std::min(remaining_rows, remaining_capacity);
		if (append_count == 0) {
			return;
		}

		auto output_row = batch.row_count;
		vector<int64_t> join_scratch;
		const int64_t *join_source = nullptr;
		if (InferGridFromRowOrder()) {
			join_scratch.resize(append_count);
			for (idx_t row = 0; row < append_count; row++) {
				join_scratch[row] = RowOrderJoinKey(*batch.mapping, chunk_row_base + row_offset + row);
			}
			join_source = join_scratch.data();
		} else {
			join_source = join_data + row_offset;
		}
		auto rc = pipeline.copy_grids(handle, static_cast<uint32_t>(batch.slot), static_cast<uint64_t>(output_row),
		                              join_source, static_cast<uint64_t>(append_count));
		if (rc != 0) {
			throw InvalidInputException("GPU fused device-direct grid copy failed for '%s'", batch.fact_path);
		}

		for (idx_t column = 0; column < column_count; column++) {
			auto value_data = FlatVector::GetData<double>(chunk.data[payload_offset + column]);
			auto &value_validity = FlatVector::Validity(chunk.data[payload_offset + column]);
			const uint8_t *validity_data = nullptr;
			auto all_valid = value_validity.AllValid();
			if (!all_valid) {
				validity_scratch.resize(append_count);
				CopyValidityToBytes(validity_scratch.data(), value_validity, row_offset, append_count);
				validity_data = validity_scratch.data();
			}
			rc = pipeline.copy_values(handle, static_cast<uint32_t>(batch.slot), static_cast<uint64_t>(column),
			                          static_cast<uint64_t>(output_row), value_data + row_offset, validity_data,
			                          static_cast<uint64_t>(append_count), all_valid ? 1 : 0);
			if (rc != 0) {
				throw InvalidInputException("GPU fused device-direct value copy failed for '%s'", batch.fact_path);
			}
		}
		batch.row_count += append_count;
		row_offset += append_count;
		return;
	}

	std::unique_ptr<UnifiedColumnReader<int64_t>> join_data;
	if (!InferGridFromRowOrder()) {
		join_data = make_uniq<UnifiedColumnReader<int64_t>>(chunk.data[0], count);
		if (!join_data->format.validity.AllValid()) {
			throw InvalidInputException("pipeline-device-direct requires non-null join keys");
		}
	}

	auto remaining_rows = count - row_offset;
	auto remaining_capacity = target_batch_rows - batch.row_count;
	auto append_count = std::min(remaining_rows, remaining_capacity);
	if (append_count == 0) {
		return;
	}

	auto output_row = batch.row_count;
	vector<int64_t> join_scratch;
	join_scratch.resize(append_count);
	if (InferGridFromRowOrder()) {
		for (idx_t row = 0; row < append_count; row++) {
			join_scratch[row] = RowOrderJoinKey(*batch.mapping, chunk_row_base + row_offset + row);
		}
	} else {
		CopyUnifiedValues(join_scratch.data(), *join_data, row_offset, append_count);
	}
	auto rc = pipeline.copy_grids(handle, static_cast<uint32_t>(batch.slot), static_cast<uint64_t>(output_row),
	                              join_scratch.data(), static_cast<uint64_t>(append_count));
	if (rc != 0) {
		throw InvalidInputException("GPU fused device-direct grid copy failed for '%s'", batch.fact_path);
	}

	for (idx_t column = 0; column < column_count; column++) {
		UnifiedColumnReader<double> value_data(chunk.data[payload_offset + column], count);
		const uint8_t *validity_data = nullptr;
		auto all_valid = value_data.format.validity.AllValid();
		vector<double> value_scratch;
		value_scratch.resize(append_count);
		CopyUnifiedValues(value_scratch.data(), value_data, row_offset, append_count);
		if (!all_valid) {
			validity_scratch.resize(append_count);
			CopyUnifiedValidityToBytes(validity_scratch.data(), value_data, row_offset, append_count);
			validity_data = validity_scratch.data();
		}
		rc = pipeline.copy_values(handle, static_cast<uint32_t>(batch.slot), static_cast<uint64_t>(column),
		                          static_cast<uint64_t>(output_row), value_scratch.data(), validity_data,
		                          static_cast<uint64_t>(append_count), all_valid ? 1 : 0);
		if (rc != 0) {
			throw InvalidInputException("GPU fused device-direct value copy failed for '%s'", batch.fact_path);
		}
	}
	batch.row_count += append_count;
	row_offset += append_count;
}

static void AppendDirectMultiPreparedChunkRows(DirectMultiPipelineBuffer &batch, DataChunk &chunk, idx_t chunk_row_base,
                                              idx_t &row_offset, idx_t column_count, idx_t target_batch_rows) {
	auto count = chunk.size();
	auto payload_offset = MultiPayloadColumnOffset();
	if (!UseRawGpuFetch()) {
		const int64_t *join_data = nullptr;
		ValidityMask *join_validity = nullptr;
		if (!InferGridFromRowOrder()) {
			join_data = FlatVector::GetData<int64_t>(chunk.data[0]);
			join_validity = &FlatVector::Validity(chunk.data[0]);
		}

		vector<const double *> value_data;
		vector<ValidityMask *> value_validity;
		value_data.reserve(column_count);
		value_validity.reserve(column_count);
		for (idx_t column = 0; column < column_count; column++) {
			value_data.push_back(FlatVector::GetData<double>(chunk.data[payload_offset + column]));
			value_validity.push_back(&FlatVector::Validity(chunk.data[payload_offset + column]));
		}

		if (InferGridFromRowOrder() || join_validity->AllValid()) {
			auto remaining_rows = count - row_offset;
			auto remaining_capacity = target_batch_rows - batch.row_count;
			auto append_count = std::min(remaining_rows, remaining_capacity);
			if (append_count == 0) {
				return;
			}

			auto output_row = batch.row_count;
			if (InferGridFromRowOrder()) {
				for (idx_t row = 0; row < append_count; row++) {
					batch.join_keys[output_row + row] =
					    RowOrderJoinKey(*batch.mapping, chunk_row_base + row_offset + row);
				}
			} else {
				std::memcpy(batch.join_keys + output_row, join_data + row_offset, append_count * sizeof(int64_t));
			}
			for (idx_t column = 0; column < column_count; column++) {
				auto output_offset = column * batch.value_stride + output_row;
				std::memcpy(batch.values + output_offset, value_data[column] + row_offset,
				            append_count * sizeof(double));
				CopyValidityToBytes(batch.validity + output_offset, *value_validity[column], row_offset,
				                    append_count);
			}
			batch.row_count += append_count;
			row_offset += append_count;
			return;
		}

		while (row_offset < count && batch.row_count < target_batch_rows) {
			auto row = row_offset++;
			if (!join_validity->RowIsValid(row)) {
				continue;
			}

			auto output_row = batch.row_count;
			batch.join_keys[output_row] = join_data[row];
			for (idx_t column = 0; column < column_count; column++) {
				auto offset = column * batch.value_stride + output_row;
				batch.values[offset] = value_data[column][row];
				batch.validity[offset] = value_validity[column]->RowIsValid(row) ? 1 : 0;
			}
			batch.row_count++;
		}
		return;
	}

	std::unique_ptr<UnifiedColumnReader<int64_t>> join_data;
	if (!InferGridFromRowOrder()) {
		join_data = make_uniq<UnifiedColumnReader<int64_t>>(chunk.data[0], count);
	}

	vector<UnifiedColumnReader<double>> value_data;
	value_data.reserve(column_count);
	for (idx_t column = 0; column < column_count; column++) {
		value_data.emplace_back(chunk.data[payload_offset + column], count);
	}

	if (InferGridFromRowOrder() || join_data->format.validity.AllValid()) {
		auto remaining_rows = count - row_offset;
		auto remaining_capacity = target_batch_rows - batch.row_count;
		auto append_count = std::min(remaining_rows, remaining_capacity);
		if (append_count == 0) {
			return;
		}

		auto output_row = batch.row_count;
		if (InferGridFromRowOrder()) {
			for (idx_t row = 0; row < append_count; row++) {
				batch.join_keys[output_row + row] =
				    RowOrderJoinKey(*batch.mapping, chunk_row_base + row_offset + row);
			}
		} else {
			CopyUnifiedValues(batch.join_keys + output_row, *join_data, row_offset, append_count);
		}
		for (idx_t column = 0; column < column_count; column++) {
			auto output_offset = column * batch.value_stride + output_row;
			CopyUnifiedValues(batch.values + output_offset, value_data[column], row_offset, append_count);
			CopyUnifiedValidityToBytes(batch.validity + output_offset, value_data[column], row_offset, append_count);
		}
		batch.row_count += append_count;
		row_offset += append_count;
		return;
	}

	while (row_offset < count && batch.row_count < target_batch_rows) {
		auto row = row_offset++;
		if (!join_data->RowIsValid(row)) {
			continue;
		}

		auto output_row = batch.row_count;
		batch.join_keys[output_row] = join_data->Value(row);
		for (idx_t column = 0; column < column_count; column++) {
			auto offset = column * batch.value_stride + output_row;
			batch.values[offset] = value_data[column].Value(row);
			batch.validity[offset] = value_data[column].RowIsValid(row) ? 1 : 0;
		}
		batch.row_count++;
	}
}

static ProbeColumns ReadProbeColumns(Connection &connection, const string &fact_path, const string &join_key,
                                     const string &payload_column) {
	auto result = RunStreamingQuery(connection, BuildProbeQuery(fact_path, join_key, payload_column));
	ProbeColumns columns;
	ProbeColumns chunk_columns;

	while (true) {
		auto chunk = FetchGpuPipelineChunk(*result);
		if (!chunk || chunk->size() == 0) {
			break;
		}
		AppendProbeChunk(*chunk, chunk_columns);
		columns.join_keys.insert(columns.join_keys.end(), chunk_columns.join_keys.begin(), chunk_columns.join_keys.end());
		columns.values.insert(columns.values.end(), chunk_columns.values.begin(), chunk_columns.values.end());
		columns.validity.insert(columns.validity.end(), chunk_columns.validity.begin(), chunk_columns.validity.end());
	}
	return columns;
}

static string BuildMultiProbeQuery(const string &fact_path, const string &join_key,
                                   const vector<string> &payload_columns) {
	string query = "SELECT ";
	if (!InferGridFromRowOrder()) {
		query += QuoteIdentifier(join_key);
		query += " AS join_key";
	}
	for (idx_t column = 0; column < payload_columns.size(); column++) {
		auto quoted = QuoteIdentifier(payload_columns[column]);
		if (column > 0 || !InferGridFromRowOrder()) {
			query += ", ";
		}
		query += quoted + " AS value_" + std::to_string(column);
	}
	query += " FROM " + BuildReadParquetExpression(fact_path);
	return query;
}

static void AppendMultiProbeChunk(DataChunk &chunk, MultiProbeColumns &columns, idx_t column_count) {
	auto count = chunk.size();
	UnifiedColumnReader<int64_t> join_data(chunk.data[0], count);
	vector<UnifiedColumnReader<double>> value_data;
	value_data.reserve(column_count);
	for (idx_t column = 0; column < column_count; column++) {
		value_data.emplace_back(chunk.data[1 + column], count);
	}

	if (columns.values.empty()) {
		columns.values.resize(column_count);
		columns.validity.resize(column_count);
	}
	for (idx_t column = 0; column < column_count; column++) {
		columns.values[column].reserve(columns.values[column].size() + count);
		columns.validity[column].reserve(columns.validity[column].size() + count);
	}
	columns.join_keys.reserve(columns.join_keys.size() + count);

	for (idx_t row = 0; row < count; row++) {
		if (!join_data.RowIsValid(row)) {
			continue;
		}
		columns.join_keys.push_back(join_data.Value(row));
		for (idx_t column = 0; column < column_count; column++) {
			columns.values[column].push_back(value_data[column].Value(row));
			columns.validity[column].push_back(value_data[column].RowIsValid(row) ? 1 : 0);
		}
	}
}

static MultiProbeColumns ReadMultiProbeColumns(Connection &connection, const string &fact_path, const string &join_key,
                                               const vector<string> &payload_columns) {
	auto result = RunStreamingQuery(connection, BuildMultiProbeQuery(fact_path, join_key, payload_columns));
	MultiProbeColumns columns;
	columns.values.resize(payload_columns.size());
	columns.validity.resize(payload_columns.size());

	while (true) {
		auto chunk = FetchGpuPipelineChunk(*result);
		if (!chunk || chunk->size() == 0) {
			break;
		}
		AppendMultiProbeChunk(*chunk, columns, payload_columns.size());
	}
	return columns;
}

static void FlattenMultiProbeColumns(const MultiProbeColumns &columns, vector<double> &values,
                                     vector<uint8_t> &validity) {
	auto row_count = columns.join_keys.size();
	auto column_count = columns.values.size();
	values.resize(column_count * row_count);
	validity.resize(column_count * row_count);
	for (idx_t column = 0; column < column_count; column++) {
		std::memcpy(values.data() + column * row_count, columns.values[column].data(), row_count * sizeof(double));
		std::memcpy(validity.data() + column * row_count, columns.validity[column].data(),
		            row_count * sizeof(uint8_t));
	}
}

static void MergeFusedResult(std::map<double, double> &total_sum, std::map<double, uint64_t> &total_count,
                             uint64_t &total_rows, const GroupMapping &mapping, const vector<double> &sums,
                             const vector<uint64_t> &counts, const vector<uint64_t> &row_counts) {
	auto group_count = mapping.group_values.size();
	for (idx_t group = 0; group < group_count; group++) {
		auto row_count = row_counts[group];
		if (row_count == 0) {
			continue;
		}
		auto group_value = mapping.group_values[group];
		total_sum[group_value] += sums[group];
		total_count[group_value] += counts[group];
		total_rows += row_count;
	}
}

static void MergeMultiFusedResult(vector<std::map<double, double>> &total_sums,
                                  vector<std::map<double, uint64_t>> &total_counts,
                                  std::map<double, uint64_t> &total_row_counts, uint64_t &total_rows,
                                  const GroupMapping &mapping, const vector<double> &sums,
                                  const vector<uint64_t> &counts, const vector<uint64_t> &row_counts,
                                  idx_t column_count) {
	auto group_count = mapping.group_values.size();
	for (idx_t group = 0; group < group_count; group++) {
		auto row_count = row_counts[group];
		if (row_count == 0) {
			continue;
		}
		auto group_value = mapping.group_values[group];
		total_rows += row_count;
		total_row_counts[group_value] += row_count;
		for (idx_t column = 0; column < column_count; column++) {
			auto offset = column * group_count + group;
			total_sums[column][group_value] += sums[offset];
			total_counts[column][group_value] += counts[offset];
		}
	}
}

static idx_t GetOrCreateAccumulatedGroup(MultiPipelineAccumulatedResult &total, double group_value) {
	auto entry = total.group_index.find(group_value);
	if (entry != total.group_index.end()) {
		return entry->second;
	}

	auto group = total.group_values.size();
	total.group_index[group_value] = group;
	total.group_values.push_back(group_value);
	total.row_counts.push_back(0);
	return group;
}

static void InitializeAccumulatedResult(MultiPipelineAccumulatedResult &total, idx_t column_count) {
	if (total.column_count == 0) {
		total.column_count = column_count;
		return;
	}
	if (total.column_count != column_count) {
		throw InvalidInputException("GPU fused multi merge column count changed unexpectedly");
	}
}

static void MergeMultiFusedResultFast(MultiPipelineAccumulatedResult &total, const GroupMapping &mapping,
                                      const vector<double> &sums, const vector<uint64_t> &counts,
                                      const vector<uint64_t> &row_counts, idx_t column_count) {
	InitializeAccumulatedResult(total, column_count);
	auto source_group_count = mapping.group_values.size();
	auto old_group_count = total.group_values.size();
	vector<idx_t> target_groups(source_group_count);

	for (idx_t group = 0; group < source_group_count; group++) {
		target_groups[group] = GetOrCreateAccumulatedGroup(total, mapping.group_values[group]);
	}

	auto target_group_count = total.group_values.size();
	if (target_group_count != old_group_count) {
		vector<double> resized_sums(column_count * target_group_count, 0);
		vector<uint64_t> resized_counts(column_count * target_group_count, 0);
		for (idx_t column = 0; column < column_count; column++) {
			for (idx_t group = 0; group < old_group_count; group++) {
				resized_sums[column * target_group_count + group] = total.sums[column * old_group_count + group];
				resized_counts[column * target_group_count + group] = total.counts[column * old_group_count + group];
			}
		}
		total.sums.swap(resized_sums);
		total.counts.swap(resized_counts);
	}

	for (idx_t group = 0; group < source_group_count; group++) {
		auto row_count = row_counts[group];
		if (row_count == 0) {
			continue;
		}
		auto target_group = target_groups[group];
		total.total_rows += row_count;
		total.row_counts[target_group] += row_count;
		for (idx_t column = 0; column < column_count; column++) {
			auto source_offset = column * source_group_count + group;
			auto target_offset = column * target_group_count + target_group;
			total.sums[target_offset] += sums[source_offset];
			total.counts[target_offset] += counts[source_offset];
		}
	}
}

static vector<string> ParsePayloadColumns(const py::object &payload_columns_p) {
	vector<string> payload_columns;
	if (py::isinstance<py::str>(payload_columns_p)) {
		payload_columns.push_back(py::str(payload_columns_p));
	} else {
		py::iterable payload_iterable = payload_columns_p.cast<py::iterable>();
		for (auto item : payload_iterable) {
			payload_columns.push_back(py::str(item));
		}
	}
	if (payload_columns.empty()) {
		throw InvalidInputException("payload_columns cannot be empty");
	}
	return payload_columns;
}

static void ReadPipelineRawBatches(DuckDB &db, const vector<string> &fact_paths, const string &payload_column,
                                   const string &join_key, const string &group_column, const string &dimension_file,
                                   BlockingQueue<PipelineRawBatch> &raw_queue, std::exception_ptr &error_out,
                                   std::mutex &error_lock) {
	try {
		Connection connection(db);
		ConfigurePipelineReaderConnection(connection);
		for (auto &fact_path : fact_paths) {
			auto dimension_path = ResolveDimensionPath(fact_path, dimension_file);
			auto mapping =
			    std::make_shared<GroupMapping>(ReadGroupMappingForCurrentMode(connection, dimension_path, join_key, group_column));
			auto result = RunStreamingQuery(connection, BuildProbeQuery(fact_path, join_key, payload_column));

			while (true) {
				auto chunk = FetchGpuPipelineChunk(*result);
				if (!chunk || chunk->size() == 0) {
					break;
				}

				PipelineRawBatch batch;
				batch.fact_path = fact_path;
				batch.mapping = mapping;
				AppendRawProbeChunk(*chunk, batch);
				raw_queue.Push(std::move(batch));
			}
		}
	} catch (...) {
		std::lock_guard<std::mutex> guard(error_lock);
		if (!error_out) {
			error_out = std::current_exception();
		}
	}
	raw_queue.Close();
}

static void ReadMultiPipelineRawBatches(DuckDB &db, const vector<string> &fact_paths,
                                        const vector<string> &payload_columns, const string &join_key,
                                        const string &group_column, const string &dimension_file,
                                        BlockingQueue<MultiPipelineRawBatch> &raw_queue,
                                        std::exception_ptr &error_out, std::mutex &error_lock) {
	try {
		Connection connection(db);
		ConfigurePipelineReaderConnection(connection);
		for (auto &fact_path : fact_paths) {
			auto dimension_path = ResolveDimensionPath(fact_path, dimension_file);
			auto mapping =
			    std::make_shared<GroupMapping>(ReadGroupMappingForCurrentMode(connection, dimension_path, join_key, group_column));
			auto result = RunStreamingQuery(connection, BuildMultiProbeQuery(fact_path, join_key, payload_columns));

			while (true) {
				auto chunk = FetchGpuPipelineChunk(*result);
				if (!chunk || chunk->size() == 0) {
					break;
				}

				MultiPipelineRawBatch batch;
				batch.fact_path = fact_path;
				batch.mapping = mapping;
				AppendMultiRawProbeChunk(*chunk, batch, payload_columns.size());
				raw_queue.Push(std::move(batch));
			}
		}
	} catch (...) {
		std::lock_guard<std::mutex> guard(error_lock);
		if (!error_out) {
			error_out = std::current_exception();
		}
	}
	raw_queue.Close();
}

static void ReadMultiPipelineRawBatchWorker(DuckDB &db, BlockingQueue<string> &file_queue,
                                            const vector<string> &payload_columns, const string &join_key,
                                            const string &group_column, const string &dimension_file,
                                            BlockingQueue<MultiPipelineRawBatch> &raw_queue,
                                            std::exception_ptr &error_out, std::mutex &error_lock) {
	try {
		Connection connection(db);
		ConfigurePipelineReaderConnection(connection);
		string fact_path;
		while (file_queue.Pop(fact_path)) {
			auto dimension_path = ResolveDimensionPath(fact_path, dimension_file);
			auto mapping =
			    std::make_shared<GroupMapping>(ReadGroupMappingForCurrentMode(connection, dimension_path, join_key, group_column));
			auto result = RunStreamingQuery(connection, BuildMultiProbeQuery(fact_path, join_key, payload_columns));

			while (true) {
				auto chunk = FetchGpuPipelineChunk(*result);
				if (!chunk || chunk->size() == 0) {
					break;
				}

				MultiPipelineRawBatch batch;
				batch.fact_path = fact_path;
				batch.mapping = mapping;
				AppendMultiRawProbeChunk(*chunk, batch, payload_columns.size());
				raw_queue.Push(std::move(batch));
			}
		}
	} catch (...) {
		std::lock_guard<std::mutex> guard(error_lock);
		if (!error_out) {
			error_out = std::current_exception();
		}
	}
}

static void ReadMultiPipelineChunkBatchWorker(DuckDB &db, BlockingQueue<string> &file_queue,
                                              const vector<string> &payload_columns, const string &join_key,
                                              const string &group_column, const string &dimension_file,
                                              BlockingQueue<MultiPipelineChunkBatch> &chunk_queue,
                                              std::map<string, std::shared_ptr<GroupMapping>> &dimension_mapping_cache,
                                              std::mutex &dimension_mapping_cache_lock,
                                              std::exception_ptr &error_out, std::mutex &error_lock,
                                              MultiPipelineStageTimers *timers) {
	auto stage_start = std::chrono::steady_clock::now();
	uint64_t chunk_count = 0;
	double setup_elapsed = 0;
	double connection_elapsed = 0;
	double mapping_lock_elapsed = 0;
	double mapping_elapsed = 0;
	double query_build_elapsed = 0;
	double query_submit_elapsed = 0;
	double fetch_elapsed = 0;
	double push_elapsed = 0;
	uint64_t mapping_reads = 0;
	uint64_t mapping_reuses = 0;
	try {
		auto connection_start = std::chrono::steady_clock::now();
		Connection connection(db);
		ConfigurePipelineReaderConnection(connection);
		connection_elapsed += ElapsedSeconds(connection_start);
		auto reuse_dimension_mapping = ReadEnvFlag("DUCKDB_GPU_REUSE_DIMENSION_MAPPING", false);
		string fact_path;
		while (file_queue.Pop(fact_path)) {
			auto setup_start = std::chrono::steady_clock::now();
			auto dimension_path = ResolveDimensionPath(fact_path, dimension_file);
			auto cache_key =
			    DimensionMappingCacheKey(dimension_path, dimension_file, join_key, group_column, reuse_dimension_mapping);
			std::shared_ptr<GroupMapping> mapping;
			{
				auto mapping_lock_start = std::chrono::steady_clock::now();
				std::lock_guard<std::mutex> guard(dimension_mapping_cache_lock);
				mapping_lock_elapsed += ElapsedSeconds(mapping_lock_start);
				auto cached_mapping = dimension_mapping_cache.find(cache_key);
				if (cached_mapping == dimension_mapping_cache.end()) {
					auto mapping_start = std::chrono::steady_clock::now();
					mapping = std::make_shared<GroupMapping>(
					    ReadGroupMappingForCurrentMode(connection, dimension_path, join_key, group_column));
					mapping_elapsed += ElapsedSeconds(mapping_start);
					dimension_mapping_cache[cache_key] = mapping;
					mapping_reads++;
				} else {
					mapping = cached_mapping->second;
					mapping_reuses++;
				}
			}
			auto query_build_start = std::chrono::steady_clock::now();
			auto query = BuildMultiProbeQuery(fact_path, join_key, payload_columns);
			query_build_elapsed += ElapsedSeconds(query_build_start);
			auto query_submit_start = std::chrono::steady_clock::now();
			auto result = RunStreamingQuery(connection, query);
			query_submit_elapsed += ElapsedSeconds(query_submit_start);
			setup_elapsed += ElapsedSeconds(setup_start);

			idx_t fact_row_base = 0;
			while (true) {
				auto fetch_start = std::chrono::steady_clock::now();
				auto chunk = FetchGpuPipelineChunk(*result);
				fetch_elapsed += ElapsedSeconds(fetch_start);
				if (!chunk || chunk->size() == 0) {
					break;
				}

				MultiPipelineChunkBatch batch;
				batch.fact_path = fact_path;
				batch.mapping = mapping;
				batch.row_base = fact_row_base;
				fact_row_base += chunk->size();
				batch.chunk = std::move(chunk);
				auto push_start = std::chrono::steady_clock::now();
				chunk_queue.Push(std::move(batch));
				push_elapsed += ElapsedSeconds(push_start);
				chunk_count++;
			}
		}
	} catch (...) {
		std::lock_guard<std::mutex> guard(error_lock);
		if (!error_out) {
			error_out = std::current_exception();
		}
	}
	if (timers) {
		auto read_elapsed = ElapsedSeconds(stage_start);
		std::lock_guard<std::mutex> guard(timers->lock);
		timers->read_time += read_elapsed;
		timers->read_setup_time += setup_elapsed;
		timers->read_connection_time += connection_elapsed;
		timers->read_mapping_lock_time += mapping_lock_elapsed;
		timers->read_mapping_time += mapping_elapsed;
		timers->read_query_build_time += query_build_elapsed;
		timers->read_query_submit_time += query_submit_elapsed;
		timers->read_fetch_time += fetch_elapsed;
		timers->read_push_time += push_elapsed;
		timers->read_thread_max_time = std::max(timers->read_thread_max_time, read_elapsed);
		timers->read_chunks += chunk_count;
		timers->dimension_mapping_reads += mapping_reads;
		timers->dimension_mapping_reuses += mapping_reuses;
	}
}

static void ReadMultiPipelineChunkBatches(DuckDB &db, const vector<string> &fact_paths,
                                          const vector<string> &payload_columns, const string &join_key,
                                          const string &group_column, const string &dimension_file,
                                          BlockingQueue<MultiPipelineChunkBatch> &chunk_queue,
                                          std::exception_ptr &error_out, std::mutex &error_lock,
                                          MultiPipelineStageTimers *timers) {
	auto stage_start = std::chrono::steady_clock::now();
	uint64_t chunk_count = 0;
	double setup_elapsed = 0;
	double connection_elapsed = 0;
	double mapping_elapsed = 0;
	double query_build_elapsed = 0;
	double query_submit_elapsed = 0;
	double fetch_elapsed = 0;
	double push_elapsed = 0;
	uint64_t mapping_reads = 0;
	uint64_t mapping_reuses = 0;
	try {
		auto connection_start = std::chrono::steady_clock::now();
		Connection connection(db);
		ConfigurePipelineReaderConnection(connection);
		connection_elapsed += ElapsedSeconds(connection_start);
		std::map<string, std::shared_ptr<GroupMapping>> dimension_mapping_cache;
		auto reuse_dimension_mapping = ReadEnvFlag("DUCKDB_GPU_REUSE_DIMENSION_MAPPING", false);
		for (auto &fact_path : fact_paths) {
			auto setup_start = std::chrono::steady_clock::now();
			auto dimension_path = ResolveDimensionPath(fact_path, dimension_file);
			auto cache_key =
			    DimensionMappingCacheKey(dimension_path, dimension_file, join_key, group_column, reuse_dimension_mapping);
			std::shared_ptr<GroupMapping> mapping;
			auto cached_mapping = dimension_mapping_cache.find(cache_key);
			if (cached_mapping == dimension_mapping_cache.end()) {
				auto mapping_start = std::chrono::steady_clock::now();
				mapping =
				    std::make_shared<GroupMapping>(ReadGroupMappingForCurrentMode(connection, dimension_path, join_key, group_column));
				mapping_elapsed += ElapsedSeconds(mapping_start);
				dimension_mapping_cache[cache_key] = mapping;
				mapping_reads++;
			} else {
				mapping = cached_mapping->second;
				mapping_reuses++;
			}
			auto query_build_start = std::chrono::steady_clock::now();
			auto query = BuildMultiProbeQuery(fact_path, join_key, payload_columns);
			query_build_elapsed += ElapsedSeconds(query_build_start);
			auto query_submit_start = std::chrono::steady_clock::now();
			auto result = RunStreamingQuery(connection, query);
			query_submit_elapsed += ElapsedSeconds(query_submit_start);
			setup_elapsed += ElapsedSeconds(setup_start);

			idx_t fact_row_base = 0;
			while (true) {
				auto fetch_start = std::chrono::steady_clock::now();
				auto chunk = FetchGpuPipelineChunk(*result);
				fetch_elapsed += ElapsedSeconds(fetch_start);
				if (!chunk || chunk->size() == 0) {
					break;
				}

				MultiPipelineChunkBatch batch;
				batch.fact_path = fact_path;
				batch.mapping = mapping;
				batch.row_base = fact_row_base;
				fact_row_base += chunk->size();
				batch.chunk = std::move(chunk);
				auto push_start = std::chrono::steady_clock::now();
				chunk_queue.Push(std::move(batch));
				push_elapsed += ElapsedSeconds(push_start);
				chunk_count++;
			}
		}
	} catch (...) {
		std::lock_guard<std::mutex> guard(error_lock);
		if (!error_out) {
			error_out = std::current_exception();
		}
	}
	if (timers) {
		auto read_elapsed = ElapsedSeconds(stage_start);
		std::lock_guard<std::mutex> guard(timers->lock);
		timers->read_time += read_elapsed;
		timers->read_setup_time += setup_elapsed;
		timers->read_connection_time += connection_elapsed;
		timers->read_mapping_time += mapping_elapsed;
		timers->read_query_build_time += query_build_elapsed;
		timers->read_query_submit_time += query_submit_elapsed;
		timers->read_fetch_time += fetch_elapsed;
		timers->read_push_time += push_elapsed;
		timers->read_thread_max_time = std::max(timers->read_thread_max_time, read_elapsed);
		timers->read_chunks += chunk_count;
		timers->dimension_mapping_reads += mapping_reads;
		timers->dimension_mapping_reuses += mapping_reuses;
	}
	chunk_queue.Close();
}

static vector<idx_t> ResolveParquetPayloadColumnIds(const ParquetReader &reader, const vector<string> &payload_columns,
                                                    const string &fact_path) {
	vector<idx_t> column_ids;
	column_ids.reserve(payload_columns.size());
	auto &columns = reader.GetColumns();
	for (auto &payload_column : payload_columns) {
		idx_t column_id = DConstants::INVALID_INDEX;
		for (idx_t idx = 0; idx < columns.size(); idx++) {
			if (columns[idx].name == payload_column) {
				column_id = idx;
				break;
			}
		}
		if (column_id == DConstants::INVALID_INDEX) {
			throw InvalidInputException("Parquet file '%s' does not contain payload column '%s'", fact_path,
			                            payload_column);
		}
		if (columns[column_id].type.id() != LogicalTypeId::DOUBLE) {
			throw InvalidInputException("Payload column '%s' must be DOUBLE for direct parquet GPU decode, got %s",
			                            payload_column, columns[column_id].type.ToString());
		}
		column_ids.push_back(column_id);
	}
	return column_ids;
}

static void ConfigureDirectParquetReaderProjection(ParquetReader &reader, const vector<idx_t> &column_ids) {
	reader.column_ids = MultiFileLocalColumnIds<MultiFileLocalColumnId>();
	reader.column_indexes.clear();
	for (auto column_id : column_ids) {
		reader.column_ids.emplace_back(MultiFileLocalColumnId(column_id));
		reader.column_indexes.emplace_back(ColumnIndex(column_id));
	}
}

static void MakeDirectMappedParquetOutputChunk(DataChunk &result, DirectMultiPipelineBuffer &buffer, idx_t output_row,
                                               idx_t column_count) {
	vector<LogicalType> types;
	types.reserve(column_count);
	for (idx_t column = 0; column < column_count; column++) {
		types.emplace_back(LogicalType::DOUBLE);
	}
	result.InitializeEmpty(types);
	for (idx_t column = 0; column < column_count; column++) {
		auto value_ptr = buffer.values + column * buffer.value_stride + output_row;
		FlatVector::SetData(result.data[column], reinterpret_cast<data_ptr_t>(value_ptr));
	}
	result.SetCapacity(STANDARD_VECTOR_SIZE);
}

static void FinishDirectMappedDecodedChunk(DirectMultiPipelineBuffer &buffer, DataChunk &chunk, idx_t chunk_row_base,
                                           idx_t output_row, idx_t column_count) {
	auto count = chunk.size();
	for (idx_t row = 0; row < count; row++) {
		buffer.join_keys[output_row + row] = RowOrderJoinKey(*buffer.mapping, chunk_row_base + row);
	}
	for (idx_t column = 0; column < column_count; column++) {
		auto validity_out = buffer.validity + column * buffer.value_stride + output_row;
		auto &validity = FlatVector::Validity(chunk.data[column]);
		if (validity.AllValid()) {
			std::memset(validity_out, 1, count);
		} else {
			CopyValidityToBytes(validity_out, validity, 0, count);
		}
	}
	buffer.row_count += count;
	buffer.chunks++;
}

static void FlushDirectMappedDecodedBatch(DirectMultiPipelineBuffer &current,
                                          BlockingQueue<DirectMultiPipelineInputBatch> &input_queue,
                                          BlockingQueue<idx_t> &free_slots, uint64_t &batch_count,
                                          double &push_elapsed) {
	if (!current.active) {
		current = DirectMultiPipelineBuffer();
		return;
	}
	if (current.row_count == 0) {
		free_slots.Push(current.slot);
		current = DirectMultiPipelineBuffer();
		return;
	}
	DirectMultiPipelineInputBatch batch;
	batch.fact_path = current.fact_path;
	batch.mapping = current.mapping;
	batch.slot = current.slot;
	batch.row_count = current.row_count;
	batch.value_stride = current.value_stride;
	batch.column_count = current.column_count;
	auto push_start = std::chrono::steady_clock::now();
	input_queue.Push(std::move(batch));
	push_elapsed += ElapsedSeconds(push_start);
	batch_count++;
	current = DirectMultiPipelineBuffer();
}

static void ReadDirectMappedParquetPipelineWorker(FusedLatAggMultiDirectPipelineFuncs pipeline, void *handle,
                                                  BlockingQueue<string> &file_queue,
                                                  BlockingQueue<DirectMultiPipelineInputBatch> &input_queue,
                                                  BlockingQueue<idx_t> &free_slots, DuckDB &db,
                                                  const vector<string> &payload_columns, const string &join_key,
                                                  const string &group_column, const string &dimension_file,
                                                  std::map<string, std::shared_ptr<GroupMapping>> &dimension_mapping_cache,
                                                  std::mutex &dimension_mapping_cache_lock,
                                                  std::exception_ptr &error_out, std::mutex &error_lock,
                                                  MultiPipelineStageTimers *timers) {
	auto stage_start = std::chrono::steady_clock::now();
	uint64_t chunk_count = 0;
	uint64_t batch_count = 0;
	uint64_t mapping_reads = 0;
	uint64_t mapping_reuses = 0;
	double setup_elapsed = 0;
	double connection_elapsed = 0;
	double mapping_lock_elapsed = 0;
	double mapping_elapsed = 0;
	double query_build_elapsed = 0;
	double query_submit_elapsed = 0;
	double fetch_elapsed = 0;
	double push_elapsed = 0;
	double prepare_work_elapsed = 0;
	double prepare_pop_elapsed = 0;
	double prepare_push_elapsed = 0;
	try {
		if (!InferGridFromRowOrder()) {
			throw InvalidInputException("direct parquet decode requires row-order grid inference");
		}
		auto target_batch_rows = ReadEnvIdx("DUCKDB_GPU_PIPELINE_BATCH_ROWS", 65536);
		auto target_batch_chunks = ReadEnvIdx("DUCKDB_GPU_PIPELINE_BATCH_CHUNKS", 32);
		if (target_batch_rows < STANDARD_VECTOR_SIZE || target_batch_chunks == 0) {
			throw InvalidInputException(
			    "direct parquet decode requires DUCKDB_GPU_PIPELINE_BATCH_ROWS >= STANDARD_VECTOR_SIZE and chunks > 0");
		}

		auto connection_start = std::chrono::steady_clock::now();
		Connection connection(db);
		ConfigurePipelineReaderConnection(connection);
		auto &context = *connection.context;
		connection_elapsed += ElapsedSeconds(connection_start);

		string fact_path;
		while (file_queue.Pop(fact_path)) {
			auto setup_start = std::chrono::steady_clock::now();
			auto dimension_path = ResolveDimensionPath(fact_path, dimension_file);
			auto cache_key =
			    DimensionMappingCacheKey(dimension_path, dimension_file, join_key, group_column, true);
			std::shared_ptr<GroupMapping> mapping;
			{
				auto mapping_lock_start = std::chrono::steady_clock::now();
				std::lock_guard<std::mutex> guard(dimension_mapping_cache_lock);
				mapping_lock_elapsed += ElapsedSeconds(mapping_lock_start);
				auto cached_mapping = dimension_mapping_cache.find(cache_key);
				if (cached_mapping == dimension_mapping_cache.end()) {
					auto mapping_start = std::chrono::steady_clock::now();
					mapping = std::make_shared<GroupMapping>(
					    ReadGroupMappingForCurrentMode(connection, dimension_path, join_key, group_column));
					mapping_elapsed += ElapsedSeconds(mapping_start);
					dimension_mapping_cache[cache_key] = mapping;
					mapping_reads++;
				} else {
					mapping = cached_mapping->second;
					mapping_reuses++;
				}
			}

			auto query_build_start = std::chrono::steady_clock::now();
			ParquetOptions parquet_options(context);
			query_build_elapsed += ElapsedSeconds(query_build_start);
			auto query_submit_start = std::chrono::steady_clock::now();
			ParquetReader reader(context, OpenFileInfo(fact_path), parquet_options);
			auto projected_column_ids = ResolveParquetPayloadColumnIds(reader, payload_columns, fact_path);
			ConfigureDirectParquetReaderProjection(reader, projected_column_ids);
			ParquetReaderScanState scan_state;
			vector<idx_t> groups_to_read;
			groups_to_read.reserve(reader.NumRowGroups());
			for (idx_t group = 0; group < reader.NumRowGroups(); group++) {
				groups_to_read.push_back(group);
			}
			reader.InitializeScan(context, scan_state, std::move(groups_to_read));
			query_submit_elapsed += ElapsedSeconds(query_submit_start);
			setup_elapsed += ElapsedSeconds(setup_start);

			DirectMultiPipelineBuffer current;
			idx_t fact_row_base = 0;
			while (true) {
				if (!current.active || current.chunks >= target_batch_chunks ||
				    target_batch_rows - current.row_count < STANDARD_VECTOR_SIZE) {
					FlushDirectMappedDecodedBatch(current, input_queue, free_slots, batch_count, prepare_push_elapsed);
					auto prepare_start = std::chrono::steady_clock::now();
					MultiPipelineChunkBatch start_batch;
					start_batch.fact_path = fact_path;
					start_batch.mapping = mapping;
					StartDirectMultiPipelineBuffer(pipeline, handle, free_slots, target_batch_rows,
					                               payload_columns.size(), start_batch, current, false);
					prepare_work_elapsed += ElapsedSeconds(prepare_start);
				}

				auto output_row = current.row_count;
				DataChunk result;
				MakeDirectMappedParquetOutputChunk(result, current, output_row, payload_columns.size());
				auto fetch_start = std::chrono::steady_clock::now();
				auto scan_result = reader.Scan(context, scan_state, result);
				if (scan_result.GetResultType() == AsyncResultType::BLOCKED) {
					scan_result.ExecuteTasksSynchronously();
				}
				fetch_elapsed += ElapsedSeconds(fetch_start);
				if (scan_result.GetResultType() == AsyncResultType::FINISHED) {
					break;
				}
				if (result.size() == 0) {
					continue;
				}
				auto work_start = std::chrono::steady_clock::now();
				FinishDirectMappedDecodedChunk(current, result, fact_row_base, output_row, payload_columns.size());
				prepare_work_elapsed += ElapsedSeconds(work_start);
				fact_row_base += result.size();
				chunk_count++;
			}
			FlushDirectMappedDecodedBatch(current, input_queue, free_slots, batch_count, prepare_push_elapsed);
		}
	} catch (...) {
		std::lock_guard<std::mutex> guard(error_lock);
		if (!error_out) {
			error_out = std::current_exception();
		}
	}
	if (timers) {
		auto read_elapsed = ElapsedSeconds(stage_start);
		std::lock_guard<std::mutex> guard(timers->lock);
		timers->read_time += read_elapsed;
		timers->read_setup_time += setup_elapsed;
		timers->read_connection_time += connection_elapsed;
		timers->read_mapping_lock_time += mapping_lock_elapsed;
		timers->read_mapping_time += mapping_elapsed;
		timers->read_query_build_time += query_build_elapsed;
		timers->read_query_submit_time += query_submit_elapsed;
		timers->read_fetch_time += fetch_elapsed;
		timers->read_push_time += push_elapsed;
		timers->read_thread_max_time = std::max(timers->read_thread_max_time, read_elapsed);
		timers->read_chunks += chunk_count;
		timers->prepare_time += read_elapsed;
		timers->prepare_pop_time += prepare_pop_elapsed;
		timers->prepare_work_time += prepare_work_elapsed;
		timers->prepare_push_time += prepare_push_elapsed;
		timers->prepared_batches += batch_count;
		timers->dimension_mapping_reads += mapping_reads;
		timers->dimension_mapping_reuses += mapping_reuses;
	}
}

static void PrepareMultiPipelineChunkBatches(BlockingQueue<MultiPipelineChunkBatch> &chunk_queue,
                                             BlockingQueue<MultiPipelineInputBatch> &input_queue,
                                             idx_t column_count, std::exception_ptr &error_out,
                                             std::mutex &error_lock, MultiPipelineStageTimers *timers) {
	auto stage_start = std::chrono::steady_clock::now();
	uint64_t batch_count = 0;
	double pop_elapsed = 0;
	double work_elapsed = 0;
	double push_elapsed = 0;
	try {
		auto target_batch_rows = ReadEnvIdx("DUCKDB_GPU_PIPELINE_BATCH_ROWS", 65536);
		auto target_batch_chunks = ReadEnvIdx("DUCKDB_GPU_PIPELINE_BATCH_CHUNKS", 32);
		if (target_batch_rows == 0 || target_batch_chunks == 0) {
			throw InvalidInputException(
			    "DUCKDB_GPU_PIPELINE_BATCH_ROWS and DUCKDB_GPU_PIPELINE_BATCH_CHUNKS must be > 0");
		}
		MultiPipelineInputBatch current;
		idx_t current_chunks = 0;

		auto flush_current = [&]() {
			if (PreparedRowCount(current) > 0) {
				auto push_start = std::chrono::steady_clock::now();
				input_queue.Push(std::move(current));
				push_elapsed += ElapsedSeconds(push_start);
				batch_count++;
			}
			current = MultiPipelineInputBatch();
			current_chunks = 0;
		};

		MultiPipelineChunkBatch chunk_batch;
		while (true) {
			auto pop_start = std::chrono::steady_clock::now();
			auto has_chunk = chunk_queue.Pop(chunk_batch);
			pop_elapsed += ElapsedSeconds(pop_start);
			if (!has_chunk) {
				break;
			}
			if (!chunk_batch.chunk || chunk_batch.chunk->size() == 0) {
				continue;
			}
			idx_t row_offset = 0;
			while (row_offset < chunk_batch.chunk->size()) {
				auto boundary = !current.fact_path.empty() && current.mapping != chunk_batch.mapping;
				if (boundary || (PreparedRowCount(current) > 0 && current_chunks >= target_batch_chunks)) {
					flush_current();
				}
				if (current.fact_path.empty()) {
					StartMultiPipelineInputBatch(current, chunk_batch.fact_path, chunk_batch.mapping, column_count,
					                             target_batch_rows);
				}
				if (current_chunks == 0 || row_offset == 0) {
					current_chunks++;
				}

				auto work_start = std::chrono::steady_clock::now();
				AppendMultiPreparedChunkRows(current, *chunk_batch.chunk, chunk_batch.row_base, row_offset,
				                             column_count, target_batch_rows);
				work_elapsed += ElapsedSeconds(work_start);
				if (PreparedRowCount(current) >= target_batch_rows) {
					flush_current();
				}
			}
		}
		flush_current();
	} catch (...) {
		std::lock_guard<std::mutex> guard(error_lock);
		if (!error_out) {
			error_out = std::current_exception();
		}
	}
	if (timers) {
		std::lock_guard<std::mutex> guard(timers->lock);
		timers->prepare_time += ElapsedSeconds(stage_start);
		timers->prepare_pop_time += pop_elapsed;
		timers->prepare_work_time += work_elapsed;
		timers->prepare_push_time += push_elapsed;
		timers->prepared_batches += batch_count;
	}
	input_queue.Close();
}

static void PrepareDirectMultiPipelineChunkBatches(FusedLatAggMultiDirectPipelineFuncs pipeline, void *handle,
                                                   BlockingQueue<MultiPipelineChunkBatch> &chunk_queue,
                                                   BlockingQueue<DirectMultiPipelineInputBatch> &input_queue,
                                                   BlockingQueue<idx_t> &free_slots, idx_t column_count,
                                                   std::exception_ptr &error_out, std::mutex &error_lock,
                                                   MultiPipelineStageTimers *timers, bool device_direct = false) {
	auto stage_start = std::chrono::steady_clock::now();
	uint64_t batch_count = 0;
	double pop_elapsed = 0;
	double work_elapsed = 0;
	double push_elapsed = 0;
	try {
		auto target_batch_rows = ReadEnvIdx("DUCKDB_GPU_PIPELINE_BATCH_ROWS", 65536);
		auto target_batch_chunks = ReadEnvIdx("DUCKDB_GPU_PIPELINE_BATCH_CHUNKS", 32);
		if (target_batch_rows == 0 || target_batch_chunks == 0) {
			throw InvalidInputException(
			    "DUCKDB_GPU_PIPELINE_BATCH_ROWS and DUCKDB_GPU_PIPELINE_BATCH_CHUNKS must be > 0");
		}
		DirectMultiPipelineBuffer current;
		vector<uint8_t> validity_scratch;

		auto flush_current = [&]() {
			if (!current.active) {
				return;
			}
			if (current.row_count == 0) {
				free_slots.Push(current.slot);
				current = DirectMultiPipelineBuffer();
				return;
			}

			DirectMultiPipelineInputBatch batch;
			batch.fact_path = current.fact_path;
			batch.mapping = current.mapping;
			batch.slot = current.slot;
			batch.row_count = current.row_count;
			batch.value_stride = current.value_stride;
			batch.column_count = current.column_count;
			auto push_start = std::chrono::steady_clock::now();
			input_queue.Push(std::move(batch));
			push_elapsed += ElapsedSeconds(push_start);
			batch_count++;
			current = DirectMultiPipelineBuffer();
		};

		MultiPipelineChunkBatch chunk_batch;
		while (true) {
			auto pop_start = std::chrono::steady_clock::now();
			auto has_chunk = chunk_queue.Pop(chunk_batch);
			pop_elapsed += ElapsedSeconds(pop_start);
			if (!has_chunk) {
				break;
			}
			if (!chunk_batch.chunk || chunk_batch.chunk->size() == 0) {
				continue;
			}

			idx_t row_offset = 0;
			bool counted_chunk = false;
			while (row_offset < chunk_batch.chunk->size()) {
				auto boundary = current.active && current.mapping != chunk_batch.mapping;
				if (boundary || current.row_count >= target_batch_rows || current.chunks >= target_batch_chunks) {
					flush_current();
					counted_chunk = false;
				}
				if (!current.active) {
					StartDirectMultiPipelineBuffer(pipeline, handle, free_slots, target_batch_rows, column_count,
					                               chunk_batch, current, device_direct);
				}
				if (!counted_chunk) {
					current.chunks++;
					counted_chunk = true;
				}

				auto work_start = std::chrono::steady_clock::now();
				if (current.device_direct) {
					AppendDeviceDirectMultiPreparedChunkRows(pipeline, handle, current, *chunk_batch.chunk,
					                                         chunk_batch.row_base, row_offset, column_count,
					                                         target_batch_rows, validity_scratch);
				} else {
					AppendDirectMultiPreparedChunkRows(current, *chunk_batch.chunk, chunk_batch.row_base, row_offset,
					                                   column_count, target_batch_rows);
				}
				work_elapsed += ElapsedSeconds(work_start);
				if (current.row_count >= target_batch_rows || current.chunks >= target_batch_chunks) {
					flush_current();
					counted_chunk = false;
				}
			}
		}
		flush_current();
	} catch (...) {
		std::lock_guard<std::mutex> guard(error_lock);
		if (!error_out) {
			error_out = std::current_exception();
		}
	}
	if (timers) {
		std::lock_guard<std::mutex> guard(timers->lock);
		timers->prepare_time += ElapsedSeconds(stage_start);
		timers->prepare_pop_time += pop_elapsed;
		timers->prepare_work_time += work_elapsed;
		timers->prepare_push_time += push_elapsed;
		timers->prepared_batches += batch_count;
	}
	input_queue.Close();
}

static void PreparePipelineInputBatches(BlockingQueue<PipelineRawBatch> &raw_queue,
                                        BlockingQueue<PipelineInputBatch> &input_queue,
                                        std::exception_ptr &error_out, std::mutex &error_lock) {
	try {
		auto target_batch_rows = ReadEnvIdx("DUCKDB_GPU_PIPELINE_BATCH_ROWS", 65536);
		auto target_batch_chunks = ReadEnvIdx("DUCKDB_GPU_PIPELINE_BATCH_CHUNKS", 32);
		PipelineInputBatch current;
		idx_t current_chunks = 0;

		auto flush_current = [&]() {
			if (PreparedRowCount(current) > 0) {
				input_queue.Push(std::move(current));
			}
			current = PipelineInputBatch();
			current_chunks = 0;
		};

		PipelineRawBatch raw;
		while (raw_queue.Pop(raw)) {
			auto raw_rows = raw.join_keys.size();
			auto boundary = !current.fact_path.empty() &&
			                (current.fact_path != raw.fact_path || current.mapping != raw.mapping);
			auto would_exceed_rows =
			    PreparedRowCount(current) > 0 && PreparedRowCount(current) + raw_rows > target_batch_rows;
			auto would_exceed_chunks = current_chunks >= target_batch_chunks;
			if (boundary || would_exceed_rows || would_exceed_chunks) {
				flush_current();
			}

			AppendPreparedRows(current, std::move(raw));
			current_chunks++;
			if (PreparedRowCount(current) >= target_batch_rows || current_chunks >= target_batch_chunks) {
				flush_current();
			}
		}
		flush_current();
	} catch (...) {
		std::lock_guard<std::mutex> guard(error_lock);
		if (!error_out) {
			error_out = std::current_exception();
		}
	}
	input_queue.Close();
}

static void PrepareMultiPipelineInputBatches(BlockingQueue<MultiPipelineRawBatch> &raw_queue,
                                             BlockingQueue<MultiPipelineInputBatch> &input_queue,
                                             idx_t column_count, std::exception_ptr &error_out,
                                             std::mutex &error_lock) {
	try {
		auto target_batch_rows = ReadEnvIdx("DUCKDB_GPU_PIPELINE_BATCH_ROWS", 65536);
		auto target_batch_chunks = ReadEnvIdx("DUCKDB_GPU_PIPELINE_BATCH_CHUNKS", 32);
		MultiPipelineInputBatch current;
		idx_t current_chunks = 0;

		auto flush_current = [&]() {
			if (PreparedRowCount(current) > 0) {
				input_queue.Push(std::move(current));
			}
			current = MultiPipelineInputBatch();
			current_chunks = 0;
		};

		MultiPipelineRawBatch raw;
		while (raw_queue.Pop(raw)) {
			auto raw_rows = raw.join_keys.size();
			auto boundary = !current.fact_path.empty() &&
			                (current.fact_path != raw.fact_path || current.mapping != raw.mapping);
			auto would_exceed_rows =
			    PreparedRowCount(current) > 0 && PreparedRowCount(current) + raw_rows > target_batch_rows;
			auto would_exceed_chunks = current_chunks >= target_batch_chunks;
			if (boundary || would_exceed_rows || would_exceed_chunks) {
				flush_current();
			}

			AppendMultiPreparedRows(current, std::move(raw), column_count, target_batch_rows);
			current_chunks++;
			if (PreparedRowCount(current) >= target_batch_rows || current_chunks >= target_batch_chunks) {
				flush_current();
			}
		}
		flush_current();
	} catch (...) {
		std::lock_guard<std::mutex> guard(error_lock);
		if (!error_out) {
			error_out = std::current_exception();
		}
	}
	input_queue.Close();
}

static void StartDirectPipelineBuffer(FusedLatAggPipelineFuncs pipeline, void *handle,
                                      BlockingQueue<idx_t> &free_slots, idx_t target_batch_rows,
                                      const PipelineRawBatch &raw, DirectPipelineBuffer &current) {
	idx_t slot = 0;
	if (!free_slots.Pop(slot)) {
		throw InvalidInputException("GPU fused direct pipeline slot queue closed");
	}

	int64_t *join_keys = nullptr;
	double *values = nullptr;
	uint8_t *validity = nullptr;
	auto rc = pipeline.prepare_input(handle, static_cast<uint32_t>(slot), static_cast<uint64_t>(target_batch_rows),
	                                 &join_keys, &values, &validity);
	if (rc != 0 || !join_keys || !values || !validity) {
		throw InvalidInputException("GPU fused direct pipeline input preparation failed for '%s'", raw.fact_path);
	}

	current = DirectPipelineBuffer();
	current.active = true;
	current.fact_path = raw.fact_path;
	current.mapping = raw.mapping;
	current.slot = slot;
	current.join_keys = join_keys;
	current.values = values;
	current.validity = validity;
}

static idx_t AppendDirectPipelineRows(PipelineRawBatch &raw, idx_t &row_offset, DirectPipelineBuffer &current,
                                      idx_t target_batch_rows) {
	auto count = raw.join_keys.size();
	idx_t appended = 0;

	while (row_offset < count && current.count < target_batch_rows) {
		if (raw.join_validity[row_offset] != 0) {
			current.join_keys[current.count] = raw.join_keys[row_offset];
			current.values[current.count] = raw.values[row_offset];
			current.validity[current.count] = raw.value_validity[row_offset];
			current.count++;
			appended++;
		}
		row_offset++;
	}
	return appended;
}

static void PrepareDirectPipelineInputBatches(FusedLatAggPipelineFuncs pipeline, void *handle,
                                             BlockingQueue<PipelineRawBatch> &raw_queue,
                                             BlockingQueue<DirectPipelineInputBatch> &input_queue,
                                             BlockingQueue<idx_t> &free_slots, std::exception_ptr &error_out,
                                             std::mutex &error_lock) {
	try {
		auto target_batch_rows = ReadEnvIdx("DUCKDB_GPU_PIPELINE_BATCH_ROWS", 65536);
		auto target_batch_chunks = ReadEnvIdx("DUCKDB_GPU_PIPELINE_BATCH_CHUNKS", 32);
		DirectPipelineBuffer current;

		auto flush_current = [&]() {
			if (!current.active) {
				return;
			}
			if (current.count == 0) {
				free_slots.Push(current.slot);
				current = DirectPipelineBuffer();
				return;
			}

			DirectPipelineInputBatch batch;
			batch.fact_path = current.fact_path;
			batch.mapping = current.mapping;
			batch.slot = current.slot;
			batch.count = current.count;
			input_queue.Push(std::move(batch));
			current = DirectPipelineBuffer();
		};

		PipelineRawBatch raw;
		while (raw_queue.Pop(raw)) {
			idx_t row_offset = 0;
			bool counted_chunk = false;
			while (row_offset < raw.join_keys.size()) {
				auto boundary = current.active &&
				                (current.fact_path != raw.fact_path || current.mapping != raw.mapping);
				if (boundary || current.count >= target_batch_rows || current.chunks >= target_batch_chunks) {
					flush_current();
					counted_chunk = false;
				}
				if (!current.active) {
					StartDirectPipelineBuffer(pipeline, handle, free_slots, target_batch_rows, raw, current);
				}
				if (!counted_chunk) {
					current.chunks++;
					counted_chunk = true;
				}

				AppendDirectPipelineRows(raw, row_offset, current, target_batch_rows);
				if (current.count >= target_batch_rows || current.chunks >= target_batch_chunks) {
					flush_current();
					counted_chunk = false;
				}
			}
		}
		flush_current();
	} catch (...) {
		std::lock_guard<std::mutex> guard(error_lock);
		if (!error_out) {
			error_out = std::current_exception();
		}
	}
	input_queue.Close();
}

static void RunPipelineGPUWorker(FusedLatAggPipelineFuncs pipeline, void *handle,
                                 BlockingQueue<PipelineInputBatch> &input_queue,
                                 BlockingQueue<PipelineOutputBatch> &output_queue, std::exception_ptr &error_out,
                                 std::mutex &error_lock) {
	try {
		constexpr idx_t ACCUMULATOR_SLOT = 0;
		PipelinePendingBatch pending;
		PipelineInputBatch batch;

		auto flush_pending = [&]() {
			if (!pending.active) {
				return;
			}
			auto rc = pipeline.wait(handle, static_cast<uint32_t>(ACCUMULATOR_SLOT), pending.sums.data(),
			                        pending.counts.data(), pending.row_counts.data());
			if (rc != 0) {
				throw InvalidInputException("GPU fused pipeline wait failed for '%s'", pending.fact_path);
			}

			PipelineOutputBatch output;
			output.fact_path = pending.fact_path;
			output.mapping = pending.mapping;
			output.sums = std::move(pending.sums);
			output.counts = std::move(pending.counts);
			output.row_counts = std::move(pending.row_counts);
			output_queue.Push(std::move(output));
			pending = PipelinePendingBatch();
		};

		auto reset_for_file = [&](PipelineInputBatch &input) {
			auto group_count = input.mapping->group_values.size();
			auto rc = pipeline.reset(handle, static_cast<uint32_t>(ACCUMULATOR_SLOT), input.mapping->join_min,
			                         input.mapping->join_max, input.mapping->join_to_group.data(),
			                         static_cast<uint64_t>(input.mapping->join_to_group.size()),
			                         static_cast<uint64_t>(group_count));
			if (rc != 0) {
				throw InvalidInputException("GPU fused pipeline accumulator reset failed for '%s'", input.fact_path);
			}
			pending.active = true;
			pending.slot = ACCUMULATOR_SLOT;
			pending.fact_path = input.fact_path;
			pending.mapping = input.mapping;
			pending.sums.assign(group_count, 0);
			pending.counts.assign(group_count, 0);
			pending.row_counts.assign(group_count, 0);
		};

		while (input_queue.Pop(batch)) {
			if (!pending.active || pending.fact_path != batch.fact_path || pending.mapping != batch.mapping) {
				flush_pending();
				reset_for_file(batch);
			}

			auto rc = pipeline.submit_accumulate(
			    handle, static_cast<uint32_t>(ACCUMULATOR_SLOT), batch.probe.join_keys.data(),
			    batch.probe.values.data(), batch.probe.validity.data(), static_cast<uint64_t>(batch.probe.join_keys.size()));
			if (rc != 0) {
				throw InvalidInputException("GPU fused pipeline accumulate submit failed for '%s'", batch.fact_path);
			}
		}

		flush_pending();
	} catch (...) {
		std::lock_guard<std::mutex> guard(error_lock);
		if (!error_out) {
			error_out = std::current_exception();
		}
	}
	output_queue.Close();
}

static void RunDirectPipelineGPUWorker(FusedLatAggPipelineFuncs pipeline, void *handle, idx_t slot_count,
                                       BlockingQueue<DirectPipelineInputBatch> &input_queue,
                                       BlockingQueue<PipelineOutputBatch> &output_queue,
                                       BlockingQueue<idx_t> &free_slots, std::exception_ptr &error_out,
                                       std::mutex &error_lock) {
	try {
		struct ActiveFile {
			bool active = false;
			string fact_path;
			std::shared_ptr<GroupMapping> mapping;
		};

		ActiveFile active_file;
		DirectPipelineInputBatch batch;

		auto flush_active_file = [&]() {
			if (!active_file.active) {
				return;
			}
			auto group_count = active_file.mapping->group_values.size();
			for (idx_t slot = 0; slot < slot_count; slot++) {
				PipelineOutputBatch output;
				output.fact_path = active_file.fact_path;
				output.mapping = active_file.mapping;
				output.sums.assign(group_count, 0);
				output.counts.assign(group_count, 0);
				output.row_counts.assign(group_count, 0);

				auto rc = pipeline.wait(handle, static_cast<uint32_t>(slot), output.sums.data(),
				                        output.counts.data(), output.row_counts.data());
				if (rc != 0) {
					throw InvalidInputException("GPU fused direct pipeline wait failed for '%s'",
					                            active_file.fact_path);
				}
				output_queue.Push(std::move(output));
			}
			active_file = ActiveFile();
		};

		auto reset_for_file = [&](const DirectPipelineInputBatch &input) {
			for (idx_t slot = 0; slot < slot_count; slot++) {
				auto rc = pipeline.reset(handle, static_cast<uint32_t>(slot), input.mapping->join_min,
				                         input.mapping->join_max, input.mapping->join_to_group.data(),
				                         static_cast<uint64_t>(input.mapping->join_to_group.size()),
				                         static_cast<uint64_t>(input.mapping->group_values.size()));
				if (rc != 0) {
					throw InvalidInputException("GPU fused direct pipeline accumulator reset failed for '%s'",
					                            input.fact_path);
				}
			}
			active_file.active = true;
			active_file.fact_path = input.fact_path;
			active_file.mapping = input.mapping;
		};

		while (input_queue.Pop(batch)) {
			if (!active_file.active || active_file.mapping != batch.mapping) {
				flush_active_file();
				reset_for_file(batch);
			}

			auto rc = pipeline.submit_prepared(handle, static_cast<uint32_t>(batch.slot),
			                                   static_cast<uint64_t>(batch.count));
			if (rc != 0) {
				throw InvalidInputException("GPU fused direct pipeline prepared submit failed for '%s'",
				                            batch.fact_path);
			}
			rc = pipeline.sync_slot(handle, static_cast<uint32_t>(batch.slot));
			if (rc != 0) {
				throw InvalidInputException("GPU fused direct pipeline slot sync failed for '%s'", batch.fact_path);
			}
			free_slots.Push(batch.slot);
		}

		flush_active_file();
	} catch (...) {
		std::lock_guard<std::mutex> guard(error_lock);
		if (!error_out) {
			error_out = std::current_exception();
		}
	}
	output_queue.Close();
	free_slots.Close();
}

static void RunPipelineCPUWorker(BlockingQueue<PipelineInputBatch> &input_queue,
                                 BlockingQueue<PipelineOutputBatch> &output_queue, std::exception_ptr &error_out,
                                 std::mutex &error_lock) {
	try {
		PipelineInputBatch batch;
		while (input_queue.Pop(batch)) {
			auto group_count = batch.mapping->group_values.size();
			PipelineOutputBatch output;
			output.fact_path = batch.fact_path;
			output.mapping = batch.mapping;
			output.sums.assign(group_count, 0);
			output.counts.assign(group_count, 0);
			output.row_counts.assign(group_count, 0);

			for (idx_t row = 0; row < batch.probe.join_keys.size(); row++) {
				auto grid = batch.probe.join_keys[row];
				if (grid < batch.mapping->join_min || grid > batch.mapping->join_max) {
					continue;
				}

				auto build_idx = static_cast<idx_t>(grid - batch.mapping->join_min);
				if (build_idx >= batch.mapping->join_to_group.size()) {
					continue;
				}

				auto group = batch.mapping->join_to_group[build_idx];
				if (group < 0 || static_cast<idx_t>(group) >= group_count) {
					continue;
				}

				output.row_counts[group]++;
				if (batch.probe.validity[row] != 0) {
					output.sums[group] += batch.probe.values[row];
					output.counts[group]++;
				}
			}

			output_queue.Push(std::move(output));
		}
	} catch (...) {
		std::lock_guard<std::mutex> guard(error_lock);
		if (!error_out) {
			error_out = std::current_exception();
		}
	}
	output_queue.Close();
}

static void RunDirectMultiPipelineGPUWorker(FusedLatAggMultiDirectPipelineFuncs pipeline, void *handle,
                                            idx_t slot_count,
                                            BlockingQueue<DirectMultiPipelineInputBatch> &input_queue,
                                            BlockingQueue<MultiPipelineOutputBatch> &output_queue,
                                            BlockingQueue<idx_t> &free_slots, idx_t column_count,
                                            std::exception_ptr &error_out, std::mutex &error_lock,
                                            MultiPipelineStageTimers *timers = nullptr) {
	auto stage_start = std::chrono::steady_clock::now();
	uint64_t batch_count = 0;
	double pop_elapsed = 0;
	double work_elapsed = 0;
	double push_elapsed = 0;
	try {
		struct ActiveFile {
			bool active = false;
			string fact_path;
			std::shared_ptr<GroupMapping> mapping;
		};

		ActiveFile active_file;
		DirectMultiPipelineInputBatch batch;

		auto flush_active_file = [&]() {
			if (!active_file.active) {
				return;
			}
			auto group_count = active_file.mapping->group_values.size();
			for (idx_t slot = 0; slot < slot_count; slot++) {
				MultiPipelineOutputBatch output;
				output.fact_path = active_file.fact_path;
				output.mapping = active_file.mapping;
				output.column_count = column_count;
				output.sums.assign(column_count * group_count, 0);
				output.counts.assign(column_count * group_count, 0);
				output.row_counts.assign(group_count, 0);

				auto rc = pipeline.wait(handle, static_cast<uint32_t>(slot), output.sums.data(),
				                        output.counts.data(), output.row_counts.data());
				if (rc != 0) {
					throw InvalidInputException("GPU fused direct multi pipeline wait failed for '%s'",
					                            active_file.fact_path);
				}
				auto push_start = std::chrono::steady_clock::now();
				output_queue.Push(std::move(output));
				push_elapsed += ElapsedSeconds(push_start);
			}
			active_file = ActiveFile();
		};

		auto reset_for_file = [&](const DirectMultiPipelineInputBatch &input) {
			for (idx_t slot = 0; slot < slot_count; slot++) {
				auto rc = pipeline.reset(handle, static_cast<uint32_t>(slot), input.mapping->join_min,
				                         input.mapping->join_max, input.mapping->join_to_group.data(),
				                         static_cast<uint64_t>(input.mapping->join_to_group.size()),
				                         static_cast<uint64_t>(input.mapping->group_values.size()),
				                         static_cast<uint64_t>(column_count));
				if (rc != 0) {
					throw InvalidInputException("GPU fused direct multi pipeline accumulator reset failed for '%s'",
					                            input.fact_path);
				}
			}
			active_file.active = true;
			active_file.fact_path = input.fact_path;
			active_file.mapping = input.mapping;
		};

		while (true) {
			auto pop_start = std::chrono::steady_clock::now();
			auto has_batch = input_queue.Pop(batch);
			pop_elapsed += ElapsedSeconds(pop_start);
			if (!has_batch) {
				break;
			}
			if (batch.row_count == 0) {
				free_slots.Push(batch.slot);
				continue;
			}
			if (!active_file.active || active_file.fact_path != batch.fact_path ||
			    active_file.mapping != batch.mapping) {
				flush_active_file();
				reset_for_file(batch);
			}

			auto gpu_start = std::chrono::steady_clock::now();
			auto rc = pipeline.submit_prepared(handle, static_cast<uint32_t>(batch.slot),
			                                   static_cast<uint64_t>(batch.row_count),
			                                   static_cast<uint64_t>(batch.column_count),
			                                   static_cast<uint64_t>(batch.value_stride));
			if (rc != 0) {
				throw InvalidInputException("GPU fused direct multi pipeline prepared submit failed for '%s'",
				                            batch.fact_path);
			}
			rc = pipeline.sync_slot(handle, static_cast<uint32_t>(batch.slot));
			if (rc != 0) {
				throw InvalidInputException("GPU fused direct multi pipeline slot sync failed for '%s'",
				                            batch.fact_path);
			}
			work_elapsed += ElapsedSeconds(gpu_start);
			batch_count++;
			free_slots.Push(batch.slot);
		}

		flush_active_file();
	} catch (...) {
		std::lock_guard<std::mutex> guard(error_lock);
		if (!error_out) {
			error_out = std::current_exception();
		}
	}
	if (timers) {
		std::lock_guard<std::mutex> guard(timers->lock);
		timers->gpu_time += ElapsedSeconds(stage_start);
		timers->gpu_pop_time += pop_elapsed;
		timers->gpu_work_time += work_elapsed;
		timers->gpu_push_time += push_elapsed;
		timers->gpu_batches += batch_count;
	}
	output_queue.Close();
	free_slots.Close();
}

static void RunMultiPipelineGPUWorker(FusedLatAggMultiStridedFunc fused_agg,
                                      BlockingQueue<MultiPipelineInputBatch> &input_queue,
                                      BlockingQueue<MultiPipelineOutputBatch> &output_queue,
                                      idx_t column_count, std::exception_ptr &error_out,
                                      std::mutex &error_lock, MultiPipelineStageTimers *timers = nullptr) {
	auto stage_start = std::chrono::steady_clock::now();
	uint64_t batch_count = 0;
	double pop_elapsed = 0;
	double work_elapsed = 0;
	double push_elapsed = 0;
	try {
		MultiPipelineInputBatch batch;
		while (true) {
			auto pop_start = std::chrono::steady_clock::now();
			auto has_batch = input_queue.Pop(batch);
			pop_elapsed += ElapsedSeconds(pop_start);
			if (!has_batch) {
				break;
			}
			auto row_count = batch.row_count;
			if (row_count == 0) {
				continue;
			}

			auto group_count = batch.mapping->group_values.size();
			MultiPipelineOutputBatch output;
			output.fact_path = batch.fact_path;
			output.mapping = batch.mapping;
			output.column_count = column_count;
			output.sums.assign(column_count * group_count, 0);
			output.counts.assign(column_count * group_count, 0);
			output.row_counts.assign(group_count, 0);

			auto gpu_start = std::chrono::steady_clock::now();
			auto rc = fused_agg(batch.join_keys.data(), batch.values.data(), batch.validity.data(),
			                    static_cast<uint64_t>(column_count), static_cast<uint64_t>(batch.value_stride),
			                    static_cast<uint64_t>(row_count), batch.mapping->join_min, batch.mapping->join_max,
			                    batch.mapping->join_to_group.data(),
			                    static_cast<uint64_t>(batch.mapping->join_to_group.size()),
			                    static_cast<uint64_t>(group_count), output.sums.data(), output.counts.data(),
			                    output.row_counts.data());
			work_elapsed += ElapsedSeconds(gpu_start);
			batch_count++;
			if (rc != 0) {
				throw InvalidInputException("GPU fused multi pipeline aggregate failed for '%s'", batch.fact_path);
			}

			auto push_start = std::chrono::steady_clock::now();
			output_queue.Push(std::move(output));
			push_elapsed += ElapsedSeconds(push_start);
		}
	} catch (...) {
		std::lock_guard<std::mutex> guard(error_lock);
		if (!error_out) {
			error_out = std::current_exception();
		}
	}
	if (timers) {
		std::lock_guard<std::mutex> guard(timers->lock);
		timers->gpu_time += ElapsedSeconds(stage_start);
		timers->gpu_pop_time += pop_elapsed;
		timers->gpu_work_time += work_elapsed;
		timers->gpu_push_time += push_elapsed;
		timers->gpu_batches += batch_count;
	}
	output_queue.Close();
}

static void RunMultiPipelineMergeWorker(BlockingQueue<MultiPipelineOutputBatch> &output_queue,
                                        MultiPipelineAccumulatedResult &total,
                                        std::exception_ptr &error_out, std::mutex &error_lock,
                                        MultiPipelineStageTimers *timers = nullptr) {
	auto stage_start = std::chrono::steady_clock::now();
	uint64_t batch_count = 0;
	double pop_elapsed = 0;
	double work_elapsed = 0;
	try {
		MultiPipelineOutputBatch output;
		while (true) {
			auto pop_start = std::chrono::steady_clock::now();
			auto has_output = output_queue.Pop(output);
			pop_elapsed += ElapsedSeconds(pop_start);
			if (!has_output) {
				break;
			}
			auto work_start = std::chrono::steady_clock::now();
			MergeMultiFusedResultFast(total, *output.mapping, output.sums, output.counts, output.row_counts,
			                          output.column_count);
			work_elapsed += ElapsedSeconds(work_start);
			batch_count++;
		}
	} catch (...) {
		std::lock_guard<std::mutex> guard(error_lock);
		if (!error_out) {
			error_out = std::current_exception();
		}
	}
	if (timers) {
		std::lock_guard<std::mutex> guard(timers->lock);
		timers->merge_time += ElapsedSeconds(stage_start);
		timers->merge_pop_time += pop_elapsed;
		timers->merge_work_time += work_elapsed;
		timers->merged_batches += batch_count;
	}
}

static void RunPipelineMergeWorker(BlockingQueue<PipelineOutputBatch> &output_queue,
                                   std::map<double, double> &total_sum,
                                   std::map<double, uint64_t> &total_count, uint64_t &total_rows,
                                   std::exception_ptr &error_out, std::mutex &error_lock) {
	try {
		PipelineOutputBatch output;
		while (output_queue.Pop(output)) {
			MergeFusedResult(total_sum, total_count, total_rows, *output.mapping, output.sums, output.counts,
			                 output.row_counts);
		}
	} catch (...) {
		std::lock_guard<std::mutex> guard(error_lock);
		if (!error_out) {
			error_out = std::current_exception();
		}
	}
}

static py::dict DBSGPUFusedLatPipeline(const py::iterable &fact_paths_p, const string &payload_column,
                                       const string &join_key, const string &group_column,
                                       const string &dimension_file, const string &lib_path, const string &mode) {
	const bool cpu_pipeline_mode = mode == "pipeline-cpu";
	const bool pipeline_direct_mode = mode == "pipeline-mapped" || mode == "pipeline-managed";
	const bool pipeline_mode = cpu_pipeline_mode || mode == "pipeline-device" || pipeline_direct_mode;
	const bool mapped = mode == "mapped" || mode == "pipeline-mapped";
	const int pipeline_memory_mode = mode == "pipeline-mapped" ? 1 : (mode == "pipeline-managed" ? 2 : 0);
	if (mode != "device" && mode != "mapped" && mode != "pipeline-device" && mode != "pipeline-mapped" &&
	    mode != "pipeline-managed" && mode != "pipeline-cpu") {
		throw InvalidInputException(
		    "mode must be 'device', 'mapped', 'pipeline-device', 'pipeline-mapped', 'pipeline-managed', or "
		    "'pipeline-cpu'");
	}
	if (InferGridFromRowOrder()) {
		throw InvalidInputException("row-order grid inference is only implemented for multi-column pipeline mode");
	}

	vector<string> fact_paths;
	for (auto item : fact_paths_p) {
		fact_paths.push_back(py::str(item));
	}
	if (fact_paths.empty()) {
		throw InvalidInputException("fact_paths cannot be empty");
	}

	auto start = std::chrono::steady_clock::now();
	DuckDB db(nullptr);
	Connection connection(db);

	std::map<double, double> total_sum;
	std::map<double, uint64_t> total_count;
	uint64_t total_rows = 0;
	ScopedThread prefetch_thread;
	if (pipeline_mode && ReadEnvFlag("DUCKDB_GPU_PREFETCH_FILES", false)) {
		prefetch_thread.Start(PrefetchPipelineFiles, fact_paths, dimension_file);
	}

	if (pipeline_mode) {
		if (cpu_pipeline_mode) {
			BlockingQueue<PipelineRawBatch> raw_queue(8);
			BlockingQueue<PipelineInputBatch> input_queue(8);
			BlockingQueue<PipelineOutputBatch> output_queue(8);
			std::exception_ptr worker_error;
			std::mutex worker_error_lock;

			std::thread reader_thread(ReadPipelineRawBatches, std::ref(db), std::cref(fact_paths),
			                          std::cref(payload_column), std::cref(join_key), std::cref(group_column),
			                          std::cref(dimension_file), std::ref(raw_queue), std::ref(worker_error),
			                          std::ref(worker_error_lock));
			std::thread prepare_thread(PreparePipelineInputBatches, std::ref(raw_queue), std::ref(input_queue),
			                           std::ref(worker_error), std::ref(worker_error_lock));
			std::thread cpu_thread(RunPipelineCPUWorker, std::ref(input_queue), std::ref(output_queue),
			                       std::ref(worker_error), std::ref(worker_error_lock));
			std::thread merge_thread(RunPipelineMergeWorker, std::ref(output_queue), std::ref(total_sum),
			                         std::ref(total_count), std::ref(total_rows), std::ref(worker_error),
			                         std::ref(worker_error_lock));

			try {
				reader_thread.join();
				prepare_thread.join();
				cpu_thread.join();
				merge_thread.join();

				if (worker_error) {
					std::rethrow_exception(worker_error);
				}
			} catch (...) {
				raw_queue.Close();
				input_queue.Close();
				output_queue.Close();
				if (reader_thread.joinable()) {
					reader_thread.join();
				}
				if (prepare_thread.joinable()) {
					prepare_thread.join();
				}
				if (cpu_thread.joinable()) {
					cpu_thread.join();
				}
				if (merge_thread.joinable()) {
					merge_thread.join();
				}
				throw;
			}
		} else {
		auto pipeline = LoadFusedLatAggPipeline(lib_path);
		constexpr idx_t PIPELINE_SLOTS = 2;
		void *handle = pipeline.create(static_cast<uint32_t>(PIPELINE_SLOTS), pipeline_memory_mode);
		if (!handle) {
			throw InvalidInputException("Failed to create GPU fused pipeline");
		}

		BlockingQueue<PipelineOutputBatch> output_queue(8);
		std::exception_ptr worker_error;
		std::mutex worker_error_lock;

		if (pipeline_direct_mode) {
			BlockingQueue<PipelineRawBatch> raw_queue(8);
			BlockingQueue<DirectPipelineInputBatch> input_queue(PIPELINE_SLOTS);
			BlockingQueue<idx_t> free_slots(0);
			for (idx_t slot = 0; slot < PIPELINE_SLOTS; slot++) {
				free_slots.Push(slot);
			}

			std::thread reader_thread(ReadPipelineRawBatches, std::ref(db), std::cref(fact_paths),
			                          std::cref(payload_column), std::cref(join_key), std::cref(group_column),
			                          std::cref(dimension_file), std::ref(raw_queue), std::ref(worker_error),
			                          std::ref(worker_error_lock));
			std::thread prepare_thread(PrepareDirectPipelineInputBatches, pipeline, handle, std::ref(raw_queue),
			                           std::ref(input_queue), std::ref(free_slots), std::ref(worker_error),
			                           std::ref(worker_error_lock));
			std::thread gpu_thread(RunDirectPipelineGPUWorker, pipeline, handle, PIPELINE_SLOTS, std::ref(input_queue),
			                       std::ref(output_queue), std::ref(free_slots), std::ref(worker_error),
			                       std::ref(worker_error_lock));
			std::thread merge_thread(RunPipelineMergeWorker, std::ref(output_queue), std::ref(total_sum),
			                         std::ref(total_count), std::ref(total_rows), std::ref(worker_error),
			                         std::ref(worker_error_lock));

			try {
				reader_thread.join();
				prepare_thread.join();
				gpu_thread.join();
				merge_thread.join();
				pipeline.destroy(handle);

				if (worker_error) {
					std::rethrow_exception(worker_error);
				}
			} catch (...) {
				raw_queue.Close();
				input_queue.Close();
				output_queue.Close();
				free_slots.Close();
				if (reader_thread.joinable()) {
					reader_thread.join();
				}
				if (prepare_thread.joinable()) {
					prepare_thread.join();
				}
				if (gpu_thread.joinable()) {
					gpu_thread.join();
				}
				if (merge_thread.joinable()) {
					merge_thread.join();
				}
				pipeline.destroy(handle);
				throw;
			}
		} else {
			BlockingQueue<PipelineRawBatch> raw_queue(8);
			BlockingQueue<PipelineInputBatch> input_queue(8);

			std::thread reader_thread(ReadPipelineRawBatches, std::ref(db), std::cref(fact_paths),
			                          std::cref(payload_column), std::cref(join_key), std::cref(group_column),
			                          std::cref(dimension_file), std::ref(raw_queue), std::ref(worker_error),
			                          std::ref(worker_error_lock));
			std::thread prepare_thread(PreparePipelineInputBatches, std::ref(raw_queue), std::ref(input_queue),
			                           std::ref(worker_error), std::ref(worker_error_lock));
			std::thread gpu_thread(RunPipelineGPUWorker, pipeline, handle, std::ref(input_queue),
			                       std::ref(output_queue), std::ref(worker_error), std::ref(worker_error_lock));
			std::thread merge_thread(RunPipelineMergeWorker, std::ref(output_queue), std::ref(total_sum),
			                         std::ref(total_count), std::ref(total_rows), std::ref(worker_error),
			                         std::ref(worker_error_lock));

			try {
				reader_thread.join();
				prepare_thread.join();
				gpu_thread.join();
				merge_thread.join();
				pipeline.destroy(handle);

				if (worker_error) {
					std::rethrow_exception(worker_error);
				}
			} catch (...) {
				raw_queue.Close();
				input_queue.Close();
				output_queue.Close();
				if (reader_thread.joinable()) {
					reader_thread.join();
				}
				if (prepare_thread.joinable()) {
					prepare_thread.join();
				}
				if (gpu_thread.joinable()) {
					gpu_thread.join();
				}
				if (merge_thread.joinable()) {
					merge_thread.join();
				}
				pipeline.destroy(handle);
				throw;
			}
		}
		}
	} else {
		auto fused_agg = LoadFusedLatAgg(lib_path, mapped);

		for (auto &fact_path : fact_paths) {
			auto dimension_path = ResolveDimensionPath(fact_path, dimension_file);
			auto mapping = ReadGroupMappingForCurrentMode(connection, dimension_path, join_key, group_column);
			auto probe = ReadProbeColumns(connection, fact_path, join_key, payload_column);
			if (probe.join_keys.empty()) {
				continue;
			}

			auto group_count = mapping.group_values.size();
			vector<double> sums(group_count, 0);
			vector<uint64_t> counts(group_count, 0);
			vector<uint64_t> row_counts(group_count, 0);

			auto rc = fused_agg(probe.join_keys.data(), probe.values.data(), probe.validity.data(),
			                    static_cast<uint64_t>(probe.join_keys.size()), mapping.join_min, mapping.join_max,
			                    mapping.join_to_group.data(), static_cast<uint64_t>(mapping.join_to_group.size()),
			                    static_cast<uint64_t>(group_count), sums.data(), counts.data(), row_counts.data());
			if (rc != 0) {
				throw InvalidInputException("GPU fused aggregate failed for '%s'", fact_path);
			}

			MergeFusedResult(total_sum, total_count, total_rows, mapping, sums, counts, row_counts);
		}
	}

	prefetch_thread.Join();
	auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
	py::dict result;
	result["row_count"] = py::int_(total_rows);
	result["input_file_count"] = py::int_(fact_paths.size());
	result["query_time"] = py::float_(elapsed);
	py::list groups;
	for (auto &entry : total_sum) {
		py::dict group;
		group["group"] = py::float_(entry.first);
		group["sum"] = py::float_(entry.second);
		group["count"] = py::int_(total_count[entry.first]);
		groups.append(group);
	}
	result["groups"] = groups;
	return result;
}

static py::dict DBSGPUFusedLatMulti(const py::iterable &fact_paths_p, const py::object &payload_columns_p,
                                    const string &join_key, const string &group_column,
                                    const string &dimension_file, const string &lib_path, const string &mode) {
	const bool pipeline_device_direct_mode = mode == "pipeline-device-direct";
	const bool pipeline_mapped_direct_mode = mode == "pipeline-mapped-direct";
	const bool pipeline_direct_mode = pipeline_device_direct_mode || pipeline_mapped_direct_mode;
	const bool pipeline_mode = mode == "pipeline-device" || mode == "pipeline-mapped" || pipeline_direct_mode;
	const bool mapped = mode == "mapped" || mode == "pipeline-mapped";
	if (mode != "device" && mode != "mapped" && mode != "pipeline-device" && mode != "pipeline-mapped" &&
	    mode != "pipeline-device-direct" && mode != "pipeline-mapped-direct") {
		throw InvalidInputException(
		    "mode must be 'device', 'mapped', 'pipeline-device', 'pipeline-mapped', 'pipeline-device-direct', "
		    "or 'pipeline-mapped-direct'");
	}
	if (InferGridFromRowOrder() && !pipeline_mode) {
		throw InvalidInputException("row-order grid inference requires a pipeline mode");
	}

	vector<string> fact_paths;
	for (auto item : fact_paths_p) {
		fact_paths.push_back(py::str(item));
	}
	if (fact_paths.empty()) {
		throw InvalidInputException("fact_paths cannot be empty");
	}
	auto payload_columns = ParsePayloadColumns(payload_columns_p);
	auto column_count = payload_columns.size();

	auto start = std::chrono::steady_clock::now();
	DuckDB db(nullptr);
	Connection connection(db);

	vector<std::map<double, double>> total_sums(column_count);
	vector<std::map<double, uint64_t>> total_counts(column_count);
	std::map<double, uint64_t> total_row_counts;
	uint64_t total_rows = 0;
	MultiPipelineAccumulatedResult accumulated_result;
	MultiPipelineStageTimers stage_timers;
	ScopedThread prefetch_thread;
	if (pipeline_mode && ReadEnvFlag("DUCKDB_GPU_PREFETCH_FILES", false)) {
		prefetch_thread.Start(PrefetchPipelineFiles, fact_paths, dimension_file);
	}

	if (pipeline_mode) {
		auto reader_thread_count = std::min<idx_t>(ReadEnvIdx("DUCKDB_GPU_PIPELINE_READER_THREADS", 1),
		                                           std::max<idx_t>(fact_paths.size(), 1));
		if (reader_thread_count == 0) {
			reader_thread_count = 1;
		}
		BlockingQueue<MultiPipelineOutputBatch> output_queue(8);
		std::exception_ptr worker_error;
		std::mutex worker_error_lock;

		if (pipeline_direct_mode) {
			constexpr idx_t PIPELINE_SLOTS = 2;
			auto direct_pipeline = LoadFusedLatAggMultiDirectPipeline(lib_path, pipeline_device_direct_mode);
			void *handle =
			    direct_pipeline.create(static_cast<uint32_t>(PIPELINE_SLOTS), pipeline_device_direct_mode ? 0 : 1);
			if (!handle) {
				throw InvalidInputException("GPU fused direct multi pipeline create failed");
			}

			BlockingQueue<MultiPipelineChunkBatch> chunk_queue(8);
			BlockingQueue<DirectMultiPipelineInputBatch> input_queue(PIPELINE_SLOTS);
			BlockingQueue<idx_t> free_slots(PIPELINE_SLOTS);
			for (idx_t slot = 0; slot < PIPELINE_SLOTS; slot++) {
				free_slots.Push(slot);
			}
			BlockingQueue<string> file_queue(0);
			std::map<string, std::shared_ptr<GroupMapping>> dimension_mapping_cache;
			std::mutex dimension_mapping_cache_lock;
			vector<std::thread> reader_threads;
			std::thread reader_thread;
			auto direct_parquet_decode =
			    pipeline_mapped_direct_mode && InferGridFromRowOrder() &&
			    ReadEnvFlag("DUCKDB_GPU_PARQUET_DIRECT_DECODE", false);
			if (direct_parquet_decode) {
				for (auto &fact_path : fact_paths) {
					file_queue.Push(fact_path);
				}
				file_queue.Close();
				reader_threads.reserve(reader_thread_count);
				for (idx_t reader = 0; reader < reader_thread_count; reader++) {
					reader_threads.emplace_back(ReadDirectMappedParquetPipelineWorker, direct_pipeline, handle,
					                            std::ref(file_queue), std::ref(input_queue), std::ref(free_slots),
					                            std::ref(db), std::cref(payload_columns), std::cref(join_key),
					                            std::cref(group_column), std::cref(dimension_file),
					                            std::ref(dimension_mapping_cache),
					                            std::ref(dimension_mapping_cache_lock), std::ref(worker_error),
					                            std::ref(worker_error_lock), &stage_timers);
				}
			} else if (reader_thread_count == 1) {
				reader_thread = std::thread(ReadMultiPipelineChunkBatches, std::ref(db), std::cref(fact_paths),
				                            std::cref(payload_columns), std::cref(join_key), std::cref(group_column),
				                            std::cref(dimension_file), std::ref(chunk_queue), std::ref(worker_error),
				                            std::ref(worker_error_lock), &stage_timers);
			} else {
				for (auto &fact_path : fact_paths) {
					file_queue.Push(fact_path);
				}
				file_queue.Close();
				reader_threads.reserve(reader_thread_count);
				for (idx_t reader = 0; reader < reader_thread_count; reader++) {
					reader_threads.emplace_back(ReadMultiPipelineChunkBatchWorker, std::ref(db), std::ref(file_queue),
					                            std::cref(payload_columns), std::cref(join_key),
					                            std::cref(group_column), std::cref(dimension_file),
					                            std::ref(chunk_queue), std::ref(dimension_mapping_cache),
					                            std::ref(dimension_mapping_cache_lock), std::ref(worker_error),
				                            std::ref(worker_error_lock), &stage_timers);
				}
			}
			std::unique_ptr<std::thread> prepare_thread;
			if (!direct_parquet_decode) {
				prepare_thread = make_uniq<std::thread>(
				    PrepareDirectMultiPipelineChunkBatches, direct_pipeline, handle, std::ref(chunk_queue),
				    std::ref(input_queue), std::ref(free_slots), column_count, std::ref(worker_error),
				    std::ref(worker_error_lock), &stage_timers, pipeline_device_direct_mode);
			}
			std::thread gpu_thread(RunDirectMultiPipelineGPUWorker, direct_pipeline, handle, PIPELINE_SLOTS,
			                       std::ref(input_queue), std::ref(output_queue), std::ref(free_slots),
			                       column_count, std::ref(worker_error), std::ref(worker_error_lock),
			                       &stage_timers);
			std::thread merge_thread(RunMultiPipelineMergeWorker, std::ref(output_queue),
			                         std::ref(accumulated_result), std::ref(worker_error),
			                         std::ref(worker_error_lock), &stage_timers);

			try {
				if (direct_parquet_decode) {
					for (auto &reader_thread_item : reader_threads) {
						reader_thread_item.join();
					}
					input_queue.Close();
				} else if (reader_thread_count == 1) {
					reader_thread.join();
				} else {
					for (auto &reader_thread_item : reader_threads) {
						reader_thread_item.join();
					}
					chunk_queue.Close();
				}
				if (prepare_thread) {
					prepare_thread->join();
				}
				gpu_thread.join();
				merge_thread.join();

				if (worker_error) {
					std::rethrow_exception(worker_error);
				}
			} catch (...) {
				chunk_queue.Close();
				input_queue.Close();
				output_queue.Close();
				free_slots.Close();
				file_queue.Close();
				if (reader_thread.joinable()) {
					reader_thread.join();
				}
				for (auto &reader_thread_item : reader_threads) {
					if (reader_thread_item.joinable()) {
						reader_thread_item.join();
					}
				}
				if (prepare_thread && prepare_thread->joinable()) {
					prepare_thread->join();
				}
				if (gpu_thread.joinable()) {
					gpu_thread.join();
				}
				if (merge_thread.joinable()) {
					merge_thread.join();
				}
				direct_pipeline.destroy(handle);
				throw;
			}
			direct_pipeline.destroy(handle);
		} else if (reader_thread_count == 1) {
			auto fused_agg_strided = LoadFusedLatAggMultiStrided(lib_path, mapped);
			BlockingQueue<MultiPipelineInputBatch> input_queue(8);
			BlockingQueue<MultiPipelineChunkBatch> chunk_queue(8);
			std::thread reader_thread(ReadMultiPipelineChunkBatches, std::ref(db), std::cref(fact_paths),
			                          std::cref(payload_columns), std::cref(join_key), std::cref(group_column),
			                          std::cref(dimension_file), std::ref(chunk_queue), std::ref(worker_error),
			                          std::ref(worker_error_lock), &stage_timers);
			std::thread prepare_thread(PrepareMultiPipelineChunkBatches, std::ref(chunk_queue), std::ref(input_queue),
			                           column_count, std::ref(worker_error), std::ref(worker_error_lock),
			                           &stage_timers);
			std::thread gpu_thread(RunMultiPipelineGPUWorker, fused_agg_strided, std::ref(input_queue),
			                       std::ref(output_queue), column_count, std::ref(worker_error),
			                       std::ref(worker_error_lock), &stage_timers);
			std::thread merge_thread(RunMultiPipelineMergeWorker, std::ref(output_queue),
			                         std::ref(accumulated_result), std::ref(worker_error),
			                         std::ref(worker_error_lock), &stage_timers);

			try {
				reader_thread.join();
				prepare_thread.join();
				gpu_thread.join();
				merge_thread.join();

				if (worker_error) {
					std::rethrow_exception(worker_error);
				}
			} catch (...) {
				chunk_queue.Close();
				input_queue.Close();
				output_queue.Close();
				if (reader_thread.joinable()) {
					reader_thread.join();
				}
				if (prepare_thread.joinable()) {
					prepare_thread.join();
				}
				if (gpu_thread.joinable()) {
					gpu_thread.join();
				}
				if (merge_thread.joinable()) {
					merge_thread.join();
				}
				throw;
			}
		} else {
			auto fused_agg_strided = LoadFusedLatAggMultiStrided(lib_path, mapped);
			BlockingQueue<MultiPipelineInputBatch> input_queue(8);
			BlockingQueue<string> file_queue(0);
			BlockingQueue<MultiPipelineChunkBatch> chunk_queue(8);
			std::map<string, std::shared_ptr<GroupMapping>> dimension_mapping_cache;
			std::mutex dimension_mapping_cache_lock;

			for (auto &fact_path : fact_paths) {
				file_queue.Push(fact_path);
			}
			file_queue.Close();

			vector<std::thread> reader_threads;
			reader_threads.reserve(reader_thread_count);
			for (idx_t reader = 0; reader < reader_thread_count; reader++) {
				reader_threads.emplace_back(ReadMultiPipelineChunkBatchWorker, std::ref(db), std::ref(file_queue),
				                            std::cref(payload_columns), std::cref(join_key), std::cref(group_column),
				                            std::cref(dimension_file), std::ref(chunk_queue),
				                            std::ref(dimension_mapping_cache),
				                            std::ref(dimension_mapping_cache_lock), std::ref(worker_error),
				                            std::ref(worker_error_lock), &stage_timers);
			}
			std::thread prepare_thread(PrepareMultiPipelineChunkBatches, std::ref(chunk_queue), std::ref(input_queue),
			                           column_count, std::ref(worker_error), std::ref(worker_error_lock),
			                           &stage_timers);
			std::thread gpu_thread(RunMultiPipelineGPUWorker, fused_agg_strided, std::ref(input_queue),
			                       std::ref(output_queue), column_count, std::ref(worker_error),
			                       std::ref(worker_error_lock), &stage_timers);
			std::thread merge_thread(RunMultiPipelineMergeWorker, std::ref(output_queue),
			                         std::ref(accumulated_result), std::ref(worker_error),
			                         std::ref(worker_error_lock), &stage_timers);

			try {
				for (auto &reader_thread : reader_threads) {
					reader_thread.join();
				}
				chunk_queue.Close();
				prepare_thread.join();
				gpu_thread.join();
				merge_thread.join();

				if (worker_error) {
					std::rethrow_exception(worker_error);
				}
			} catch (...) {
				chunk_queue.Close();
				input_queue.Close();
				output_queue.Close();
				file_queue.Close();
				for (auto &reader_thread : reader_threads) {
					if (reader_thread.joinable()) {
						reader_thread.join();
					}
				}
				if (prepare_thread.joinable()) {
					prepare_thread.join();
				}
				if (gpu_thread.joinable()) {
					gpu_thread.join();
				}
				if (merge_thread.joinable()) {
					merge_thread.join();
				}
				throw;
			}
		}
	} else {
		auto fused_agg = LoadFusedLatAggMulti(lib_path, mapped);
		for (auto &fact_path : fact_paths) {
			auto dimension_path = ResolveDimensionPath(fact_path, dimension_file);
			auto mapping = ReadGroupMappingForCurrentMode(connection, dimension_path, join_key, group_column);
			auto probe = ReadMultiProbeColumns(connection, fact_path, join_key, payload_columns);
			if (probe.join_keys.empty()) {
				continue;
			}

			vector<double> values;
			vector<uint8_t> validity;
			FlattenMultiProbeColumns(probe, values, validity);

			auto group_count = mapping.group_values.size();
			vector<double> sums(column_count * group_count, 0);
			vector<uint64_t> counts(column_count * group_count, 0);
			vector<uint64_t> row_counts(group_count, 0);

			auto rc =
			    fused_agg(probe.join_keys.data(), values.data(), validity.data(), static_cast<uint64_t>(column_count),
			              static_cast<uint64_t>(probe.join_keys.size()), mapping.join_min, mapping.join_max,
			              mapping.join_to_group.data(), static_cast<uint64_t>(mapping.join_to_group.size()),
			              static_cast<uint64_t>(group_count), sums.data(), counts.data(), row_counts.data());
			if (rc != 0) {
				throw InvalidInputException("GPU fused multi aggregate failed for '%s'", fact_path);
			}

			MergeMultiFusedResult(total_sums, total_counts, total_row_counts, total_rows, mapping, sums, counts,
			                      row_counts, column_count);
		}
	}

	if (pipeline_mode) {
		total_rows = accumulated_result.total_rows;
	}
	prefetch_thread.Join();

	auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
	py::dict result;
	result["row_count"] = py::int_(total_rows);
	result["input_file_count"] = py::int_(fact_paths.size());
	result["query_time"] = py::float_(elapsed);

	py::dict stage_times;
	{
		std::lock_guard<std::mutex> guard(stage_timers.lock);
		stage_times["read_time"] = py::float_(stage_timers.read_time);
		stage_times["prepare_time"] = py::float_(stage_timers.prepare_time);
		stage_times["gpu_time"] = py::float_(stage_timers.gpu_time);
		stage_times["merge_time"] = py::float_(stage_timers.merge_time);
		stage_times["read_setup_time"] = py::float_(stage_timers.read_setup_time);
		stage_times["read_connection_time"] = py::float_(stage_timers.read_connection_time);
		stage_times["read_mapping_lock_time"] = py::float_(stage_timers.read_mapping_lock_time);
		stage_times["read_mapping_time"] = py::float_(stage_timers.read_mapping_time);
		stage_times["read_query_build_time"] = py::float_(stage_timers.read_query_build_time);
		stage_times["read_query_submit_time"] = py::float_(stage_timers.read_query_submit_time);
		stage_times["read_fetch_time"] = py::float_(stage_timers.read_fetch_time);
		stage_times["read_push_time"] = py::float_(stage_timers.read_push_time);
		stage_times["read_thread_max_time"] = py::float_(stage_timers.read_thread_max_time);
		stage_times["prepare_pop_time"] = py::float_(stage_timers.prepare_pop_time);
		stage_times["prepare_work_time"] = py::float_(stage_timers.prepare_work_time);
		stage_times["prepare_push_time"] = py::float_(stage_timers.prepare_push_time);
		stage_times["gpu_pop_time"] = py::float_(stage_timers.gpu_pop_time);
		stage_times["gpu_work_time"] = py::float_(stage_timers.gpu_work_time);
		stage_times["gpu_push_time"] = py::float_(stage_timers.gpu_push_time);
		stage_times["merge_pop_time"] = py::float_(stage_timers.merge_pop_time);
		stage_times["merge_work_time"] = py::float_(stage_timers.merge_work_time);
		stage_times["read_chunks"] = py::int_(stage_timers.read_chunks);
		stage_times["prepared_batches"] = py::int_(stage_timers.prepared_batches);
		stage_times["gpu_batches"] = py::int_(stage_timers.gpu_batches);
		stage_times["merged_batches"] = py::int_(stage_timers.merged_batches);
		stage_times["dimension_mapping_reads"] = py::int_(stage_timers.dimension_mapping_reads);
		stage_times["dimension_mapping_reuses"] = py::int_(stage_timers.dimension_mapping_reuses);
	}
	result["stage_times"] = stage_times;

	py::list payloads;
	for (auto &payload_column : payload_columns) {
		payloads.append(payload_column);
	}
	result["payload_columns"] = payloads;

	py::list groups;
	if (pipeline_mode) {
		auto group_count = accumulated_result.group_values.size();
		for (idx_t group_idx = 0; group_idx < group_count; group_idx++) {
			auto group_value = accumulated_result.group_values[group_idx];
			py::dict group;
			group["group"] = py::float_(group_value);
			group["row_count"] = py::int_(accumulated_result.row_counts[group_idx]);
			for (idx_t column = 0; column < column_count; column++) {
				auto &payload_column = payload_columns[column];
				auto offset = column * group_count + group_idx;
				auto sum = accumulated_result.sums[offset];
				auto count = accumulated_result.counts[offset];
				group[py::str("sum_" + payload_column)] = py::float_(sum);
				group[py::str("count_" + payload_column)] = py::int_(count);
				if (count > 0) {
					group[py::str("avg_" + payload_column)] = py::float_(sum / static_cast<double>(count));
				} else {
					group[py::str("avg_" + payload_column)] = py::none();
				}
			}
			groups.append(group);
		}
	} else {
		for (auto &entry : total_row_counts) {
			auto group_value = entry.first;
			py::dict group;
			group["group"] = py::float_(group_value);
			group["row_count"] = py::int_(entry.second);
			for (idx_t column = 0; column < column_count; column++) {
				auto &payload_column = payload_columns[column];
				auto sum = total_sums[column][group_value];
				auto count = total_counts[column][group_value];
				group[py::str("sum_" + payload_column)] = py::float_(sum);
				group[py::str("count_" + payload_column)] = py::int_(count);
				if (count > 0) {
					group[py::str("avg_" + payload_column)] = py::float_(sum / static_cast<double>(count));
				} else {
					group[py::str("avg_" + payload_column)] = py::none();
				}
			}
			groups.append(group);
		}
	}
	result["groups"] = groups;
	return result;
}

} // namespace

void RegisterDBSGPUFused(py::module_ &m) {
	m.def("dbs_gpu_fused_lat_pipeline", &DBSGPUFusedLatPipeline, py::arg("fact_paths"),
	      py::arg("payload_column") = "qicps", py::arg("join_key") = "grid", py::arg("group_column") = "lats",
	      py::arg("dimension_file") = "grid.parquet", py::arg("lib_path") = "",
	      py::arg("mode") = "pipeline-managed");
	m.def("dbs_gpu_fused_lat_multi", &DBSGPUFusedLatMulti, py::arg("fact_paths"),
	      py::arg("payload_columns") = py::make_tuple("qicps"), py::arg("join_key") = "grid",
	      py::arg("group_column") = "lats", py::arg("dimension_file") = "grid.parquet", py::arg("lib_path") = "",
	      py::arg("mode") = "device");
}

} // namespace duckdb
