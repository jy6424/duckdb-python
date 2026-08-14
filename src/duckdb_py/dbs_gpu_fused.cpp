#include "duckdb_python/pybind11/pybind_wrapper.hpp"

#include "duckdb.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/vector.hpp"

#include <chrono>
#include <algorithm>
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
#include <thread>
#include <utility>

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

static string ResolveDimensionPath(const string &fact_path, const string &dimension_file) {
	if (!dimension_file.empty() && (dimension_file[0] == '/' || dimension_file.find('/') != string::npos)) {
		return dimension_file;
	}
	return ParentPath(fact_path) + "/" + dimension_file;
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

static unique_ptr<QueryResult> RunStreamingQuery(Connection &connection, const string &query) {
	auto result = connection.SendQuery(query, QueryResultOutputType::ALLOW_STREAMING);
	if (result->HasError()) {
		result->ThrowError();
	}
	return result;
}

struct GroupMapping {
	int64_t join_min = 0;
	int64_t join_max = 0;
	vector<int32_t> join_to_group;
	vector<double> group_values;
};

static GroupMapping ReadGroupMapping(Connection &connection, const string &dimension_path, const string &join_key,
                                     const string &group_column) {
	auto query = StringUtil::Format(
	    "SELECT %s::BIGINT AS join_key, %s::DOUBLE AS group_value FROM read_parquet('%s')",
	    QuoteIdentifier(join_key), QuoteIdentifier(group_column), EscapeSQLString(dimension_path));
	auto result = RunStreamingQuery(connection, query);

	vector<int64_t> join_keys;
	vector<double> group_values_per_row;
	bool first = true;
	int64_t join_min = 0;
	int64_t join_max = 0;

	while (true) {
		auto chunk = result->Fetch();
		if (!chunk || chunk->size() == 0) {
			break;
		}
		auto count = chunk->size();
		auto join_data = FlatVector::GetData<int64_t>(chunk->data[0]);
		auto group_data = FlatVector::GetData<double>(chunk->data[1]);
		auto &join_validity = FlatVector::Validity(chunk->data[0]);
		auto &group_validity = FlatVector::Validity(chunk->data[1]);
		for (idx_t row = 0; row < count; row++) {
			if (!join_validity.RowIsValid(row) || !group_validity.RowIsValid(row)) {
				continue;
			}
			auto join_value = join_data[row];
			auto group_value = group_data[row];
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

struct MultiPipelineOutputBatch {
	string fact_path;
	std::shared_ptr<GroupMapping> mapping;
	vector<double> sums;
	vector<uint64_t> counts;
	vector<uint64_t> row_counts;
	idx_t column_count = 0;
};

struct MultiPipelineStageTimers {
	std::mutex lock;
	double read_time = 0;
	double prepare_time = 0;
	double gpu_time = 0;
	double merge_time = 0;
	uint64_t read_chunks = 0;
	uint64_t prepared_batches = 0;
	uint64_t gpu_batches = 0;
	uint64_t merged_batches = 0;
};

static double ElapsedSeconds(std::chrono::steady_clock::time_point start) {
	return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

static string BuildProbeQuery(const string &fact_path, const string &join_key, const string &payload_column) {
	return StringUtil::Format(
	    "SELECT %s::BIGINT AS join_key, COALESCE(%s, 0)::DOUBLE AS value, "
	    "CASE WHEN %s IS NULL THEN 0 ELSE 1 END::UTINYINT AS value_valid FROM read_parquet('%s')",
	    QuoteIdentifier(join_key), QuoteIdentifier(payload_column), QuoteIdentifier(payload_column),
	    EscapeSQLString(fact_path));
}

static void AppendProbeChunk(DataChunk &chunk, ProbeColumns &columns) {
	auto count = chunk.size();
	auto join_data = FlatVector::GetData<int64_t>(chunk.data[0]);
	auto value_data = FlatVector::GetData<double>(chunk.data[1]);
	auto valid_data = FlatVector::GetData<uint8_t>(chunk.data[2]);
	auto &join_validity = FlatVector::Validity(chunk.data[0]);

	columns.join_keys.clear();
	columns.values.clear();
	columns.validity.clear();
	columns.join_keys.reserve(count);
	columns.values.reserve(count);
	columns.validity.reserve(count);

	for (idx_t row = 0; row < count; row++) {
		if (!join_validity.RowIsValid(row)) {
			continue;
		}
		columns.join_keys.push_back(join_data[row]);
		columns.values.push_back(value_data[row]);
		columns.validity.push_back(valid_data[row]);
	}
}

static void AppendMultiRawProbeChunk(DataChunk &chunk, MultiPipelineRawBatch &batch, idx_t column_count) {
	auto count = chunk.size();
	auto join_data = FlatVector::GetData<int64_t>(chunk.data[0]);
	auto &join_validity = FlatVector::Validity(chunk.data[0]);

	batch.join_keys.resize(count);
	batch.join_validity.resize(count);
	batch.values.resize(column_count);
	batch.validity.resize(column_count);
	for (idx_t column = 0; column < column_count; column++) {
		batch.values[column].resize(count);
		batch.validity[column].resize(count);
		auto value_data = FlatVector::GetData<double>(chunk.data[1 + column * 2]);
		auto valid_data = FlatVector::GetData<uint8_t>(chunk.data[2 + column * 2]);
		for (idx_t row = 0; row < count; row++) {
			batch.values[column][row] = value_data[row];
			batch.validity[column][row] = valid_data[row];
		}
	}

	for (idx_t row = 0; row < count; row++) {
		batch.join_keys[row] = join_data[row];
		batch.join_validity[row] = join_validity.RowIsValid(row) ? 1 : 0;
	}
}

static void AppendRawProbeChunk(DataChunk &chunk, PipelineRawBatch &batch) {
	auto count = chunk.size();
	auto join_data = FlatVector::GetData<int64_t>(chunk.data[0]);
	auto value_data = FlatVector::GetData<double>(chunk.data[1]);
	auto valid_data = FlatVector::GetData<uint8_t>(chunk.data[2]);
	auto &join_validity = FlatVector::Validity(chunk.data[0]);

	batch.join_keys.resize(count);
	batch.values.resize(count);
	batch.value_validity.resize(count);
	batch.join_validity.resize(count);

	for (idx_t row = 0; row < count; row++) {
		batch.join_keys[row] = join_data[row];
		batch.values[row] = value_data[row];
		batch.value_validity[row] = valid_data[row];
		batch.join_validity[row] = join_validity.RowIsValid(row) ? 1 : 0;
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

static void AppendMultiPreparedChunkRows(MultiPipelineInputBatch &batch, DataChunk &chunk, idx_t &row_offset,
                                         idx_t column_count, idx_t target_batch_rows) {
	auto count = chunk.size();
	auto join_data = FlatVector::GetData<int64_t>(chunk.data[0]);
	auto &join_validity = FlatVector::Validity(chunk.data[0]);

	vector<const double *> value_data;
	vector<const uint8_t *> valid_data;
	value_data.reserve(column_count);
	valid_data.reserve(column_count);
	for (idx_t column = 0; column < column_count; column++) {
		value_data.push_back(FlatVector::GetData<double>(chunk.data[1 + column * 2]));
		valid_data.push_back(FlatVector::GetData<uint8_t>(chunk.data[2 + column * 2]));
	}

	while (row_offset < count && batch.row_count < target_batch_rows) {
		auto row = row_offset++;
		if (!join_validity.RowIsValid(row)) {
			continue;
		}

		auto output_row = batch.row_count;
		batch.join_keys.push_back(join_data[row]);
		for (idx_t column = 0; column < column_count; column++) {
			auto offset = column * batch.value_stride + output_row;
			batch.values[offset] = value_data[column][row];
			batch.validity[offset] = valid_data[column][row];
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
		auto chunk = result->Fetch();
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
	query += QuoteIdentifier(join_key);
	query += "::BIGINT AS join_key";
	for (idx_t column = 0; column < payload_columns.size(); column++) {
		auto quoted = QuoteIdentifier(payload_columns[column]);
		query += ", COALESCE(" + quoted + ", 0)::DOUBLE AS value_" + std::to_string(column);
		query += ", CASE WHEN " + quoted + " IS NULL THEN 0 ELSE 1 END::UTINYINT AS valid_" +
		         std::to_string(column);
	}
	query += " FROM read_parquet('" + EscapeSQLString(fact_path) + "')";
	return query;
}

static void AppendMultiProbeChunk(DataChunk &chunk, MultiProbeColumns &columns, idx_t column_count) {
	auto count = chunk.size();
	auto join_data = FlatVector::GetData<int64_t>(chunk.data[0]);
	auto &join_validity = FlatVector::Validity(chunk.data[0]);

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
		if (!join_validity.RowIsValid(row)) {
			continue;
		}
		columns.join_keys.push_back(join_data[row]);
		for (idx_t column = 0; column < column_count; column++) {
			auto value_data = FlatVector::GetData<double>(chunk.data[1 + column * 2]);
			auto valid_data = FlatVector::GetData<uint8_t>(chunk.data[2 + column * 2]);
			columns.values[column].push_back(value_data[row]);
			columns.validity[column].push_back(valid_data[row]);
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
		auto chunk = result->Fetch();
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
		for (auto &fact_path : fact_paths) {
			auto dimension_path = ResolveDimensionPath(fact_path, dimension_file);
			auto mapping =
			    std::make_shared<GroupMapping>(ReadGroupMapping(connection, dimension_path, join_key, group_column));
			auto result = RunStreamingQuery(connection, BuildProbeQuery(fact_path, join_key, payload_column));

			while (true) {
				auto chunk = result->Fetch();
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
		for (auto &fact_path : fact_paths) {
			auto dimension_path = ResolveDimensionPath(fact_path, dimension_file);
			auto mapping =
			    std::make_shared<GroupMapping>(ReadGroupMapping(connection, dimension_path, join_key, group_column));
			auto result = RunStreamingQuery(connection, BuildMultiProbeQuery(fact_path, join_key, payload_columns));

			while (true) {
				auto chunk = result->Fetch();
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
		string fact_path;
		while (file_queue.Pop(fact_path)) {
			auto dimension_path = ResolveDimensionPath(fact_path, dimension_file);
			auto mapping =
			    std::make_shared<GroupMapping>(ReadGroupMapping(connection, dimension_path, join_key, group_column));
			auto result = RunStreamingQuery(connection, BuildMultiProbeQuery(fact_path, join_key, payload_columns));

			while (true) {
				auto chunk = result->Fetch();
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

static void ReadMultiPipelineChunkBatches(DuckDB &db, const vector<string> &fact_paths,
                                          const vector<string> &payload_columns, const string &join_key,
                                          const string &group_column, const string &dimension_file,
                                          BlockingQueue<MultiPipelineChunkBatch> &chunk_queue,
                                          std::exception_ptr &error_out, std::mutex &error_lock,
                                          MultiPipelineStageTimers *timers) {
	auto stage_start = std::chrono::steady_clock::now();
	uint64_t chunk_count = 0;
	try {
		Connection connection(db);
		for (auto &fact_path : fact_paths) {
			auto dimension_path = ResolveDimensionPath(fact_path, dimension_file);
			auto mapping =
			    std::make_shared<GroupMapping>(ReadGroupMapping(connection, dimension_path, join_key, group_column));
			auto result = RunStreamingQuery(connection, BuildMultiProbeQuery(fact_path, join_key, payload_columns));

			while (true) {
				auto chunk = result->Fetch();
				if (!chunk || chunk->size() == 0) {
					break;
				}

				MultiPipelineChunkBatch batch;
				batch.fact_path = fact_path;
				batch.mapping = mapping;
				batch.chunk = std::move(chunk);
				chunk_queue.Push(std::move(batch));
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
		std::lock_guard<std::mutex> guard(timers->lock);
		timers->read_time += ElapsedSeconds(stage_start);
		timers->read_chunks += chunk_count;
	}
	chunk_queue.Close();
}

static void PrepareMultiPipelineChunkBatches(BlockingQueue<MultiPipelineChunkBatch> &chunk_queue,
                                             BlockingQueue<MultiPipelineInputBatch> &input_queue,
                                             idx_t column_count, std::exception_ptr &error_out,
                                             std::mutex &error_lock, MultiPipelineStageTimers *timers) {
	auto stage_start = std::chrono::steady_clock::now();
	uint64_t batch_count = 0;
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
				input_queue.Push(std::move(current));
				batch_count++;
			}
			current = MultiPipelineInputBatch();
			current_chunks = 0;
		};

		MultiPipelineChunkBatch chunk_batch;
		while (chunk_queue.Pop(chunk_batch)) {
			if (!chunk_batch.chunk || chunk_batch.chunk->size() == 0) {
				continue;
			}
			idx_t row_offset = 0;
			while (row_offset < chunk_batch.chunk->size()) {
				auto boundary = !current.fact_path.empty() &&
				                (current.fact_path != chunk_batch.fact_path || current.mapping != chunk_batch.mapping);
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

				AppendMultiPreparedChunkRows(current, *chunk_batch.chunk, row_offset, column_count, target_batch_rows);
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
			if (!active_file.active || active_file.fact_path != batch.fact_path ||
			    active_file.mapping != batch.mapping) {
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

static void RunMultiPipelineGPUWorker(FusedLatAggMultiStridedFunc fused_agg,
                                      BlockingQueue<MultiPipelineInputBatch> &input_queue,
                                      BlockingQueue<MultiPipelineOutputBatch> &output_queue,
                                      idx_t column_count, std::exception_ptr &error_out,
                                      std::mutex &error_lock, MultiPipelineStageTimers *timers = nullptr) {
	uint64_t batch_count = 0;
	double gpu_elapsed = 0;
	try {
		MultiPipelineInputBatch batch;
		while (input_queue.Pop(batch)) {
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
			gpu_elapsed += ElapsedSeconds(gpu_start);
			batch_count++;
			if (rc != 0) {
				throw InvalidInputException("GPU fused multi pipeline aggregate failed for '%s'", batch.fact_path);
			}

			output_queue.Push(std::move(output));
		}
	} catch (...) {
		std::lock_guard<std::mutex> guard(error_lock);
		if (!error_out) {
			error_out = std::current_exception();
		}
	}
	if (timers) {
		std::lock_guard<std::mutex> guard(timers->lock);
		timers->gpu_time += gpu_elapsed;
		timers->gpu_batches += batch_count;
	}
	output_queue.Close();
}

static void RunMultiPipelineMergeWorker(BlockingQueue<MultiPipelineOutputBatch> &output_queue,
                                        vector<std::map<double, double>> &total_sums,
                                        vector<std::map<double, uint64_t>> &total_counts,
                                        std::map<double, uint64_t> &total_row_counts, uint64_t &total_rows,
                                        std::exception_ptr &error_out, std::mutex &error_lock,
                                        MultiPipelineStageTimers *timers = nullptr) {
	auto stage_start = std::chrono::steady_clock::now();
	uint64_t batch_count = 0;
	try {
		MultiPipelineOutputBatch output;
		while (output_queue.Pop(output)) {
			MergeMultiFusedResult(total_sums, total_counts, total_row_counts, total_rows, *output.mapping,
			                      output.sums, output.counts, output.row_counts, output.column_count);
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
			auto mapping = ReadGroupMapping(connection, dimension_path, join_key, group_column);
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
	const bool pipeline_mode = mode == "pipeline-device" || mode == "pipeline-mapped";
	const bool mapped = mode == "mapped" || mode == "pipeline-mapped";
	if (mode != "device" && mode != "mapped" && mode != "pipeline-device" && mode != "pipeline-mapped") {
		throw InvalidInputException("mode must be 'device', 'mapped', 'pipeline-device', or 'pipeline-mapped'");
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
	auto fused_agg = LoadFusedLatAggMulti(lib_path, mapped);

	vector<std::map<double, double>> total_sums(column_count);
	vector<std::map<double, uint64_t>> total_counts(column_count);
	std::map<double, uint64_t> total_row_counts;
	uint64_t total_rows = 0;
	MultiPipelineStageTimers stage_timers;

	if (pipeline_mode) {
		auto fused_agg_strided = LoadFusedLatAggMultiStrided(lib_path, mapped);
		auto reader_thread_count = std::min<idx_t>(ReadEnvIdx("DUCKDB_GPU_PIPELINE_READER_THREADS", 1),
		                                           std::max<idx_t>(fact_paths.size(), 1));
		if (reader_thread_count == 0) {
			reader_thread_count = 1;
		}
		BlockingQueue<MultiPipelineInputBatch> input_queue(8);
		BlockingQueue<MultiPipelineOutputBatch> output_queue(8);
		std::exception_ptr worker_error;
		std::mutex worker_error_lock;

		if (reader_thread_count == 1) {
			BlockingQueue<MultiPipelineChunkBatch> chunk_queue(8);
			std::thread reader_thread(ReadMultiPipelineChunkBatches, std::ref(db), std::cref(fact_paths),
			                          std::cref(payload_columns), std::cref(join_key), std::cref(group_column),
			                          std::cref(dimension_file), std::ref(chunk_queue), std::ref(worker_error),
			                          std::ref(worker_error_lock), &stage_timers);
			std::thread prepare_thread(PrepareMultiPipelineChunkBatches, std::ref(chunk_queue),
			                           std::ref(input_queue), column_count, std::ref(worker_error),
			                           std::ref(worker_error_lock), &stage_timers);
			std::thread gpu_thread(RunMultiPipelineGPUWorker, fused_agg_strided, std::ref(input_queue),
			                       std::ref(output_queue), column_count, std::ref(worker_error),
			                       std::ref(worker_error_lock), &stage_timers);
			std::thread merge_thread(RunMultiPipelineMergeWorker, std::ref(output_queue), std::ref(total_sums),
			                         std::ref(total_counts), std::ref(total_row_counts), std::ref(total_rows),
			                         std::ref(worker_error), std::ref(worker_error_lock), &stage_timers);

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
			BlockingQueue<string> file_queue(0);
			BlockingQueue<MultiPipelineRawBatch> raw_queue(8);

			for (auto &fact_path : fact_paths) {
				file_queue.Push(fact_path);
			}
			file_queue.Close();

			vector<std::thread> reader_threads;
			reader_threads.reserve(reader_thread_count);
			for (idx_t reader = 0; reader < reader_thread_count; reader++) {
				reader_threads.emplace_back(ReadMultiPipelineRawBatchWorker, std::ref(db), std::ref(file_queue),
				                            std::cref(payload_columns), std::cref(join_key), std::cref(group_column),
				                            std::cref(dimension_file), std::ref(raw_queue), std::ref(worker_error),
				                            std::ref(worker_error_lock));
			}
			std::thread prepare_thread(PrepareMultiPipelineInputBatches, std::ref(raw_queue), std::ref(input_queue),
			                           column_count, std::ref(worker_error), std::ref(worker_error_lock));
			std::thread gpu_thread(RunMultiPipelineGPUWorker, fused_agg_strided, std::ref(input_queue),
			                       std::ref(output_queue), column_count, std::ref(worker_error),
			                       std::ref(worker_error_lock), &stage_timers);
			std::thread merge_thread(RunMultiPipelineMergeWorker, std::ref(output_queue), std::ref(total_sums),
			                         std::ref(total_counts), std::ref(total_row_counts), std::ref(total_rows),
			                         std::ref(worker_error), std::ref(worker_error_lock), &stage_timers);

			try {
				for (auto &reader_thread : reader_threads) {
					reader_thread.join();
				}
				raw_queue.Close();
				prepare_thread.join();
				gpu_thread.join();
				merge_thread.join();

				if (worker_error) {
					std::rethrow_exception(worker_error);
				}
			} catch (...) {
				raw_queue.Close();
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
		for (auto &fact_path : fact_paths) {
			auto dimension_path = ResolveDimensionPath(fact_path, dimension_file);
			auto mapping = ReadGroupMapping(connection, dimension_path, join_key, group_column);
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
		stage_times["read_chunks"] = py::int_(stage_timers.read_chunks);
		stage_times["prepared_batches"] = py::int_(stage_timers.prepared_batches);
		stage_times["gpu_batches"] = py::int_(stage_timers.gpu_batches);
		stage_times["merged_batches"] = py::int_(stage_timers.merged_batches);
	}
	result["stage_times"] = stage_times;

	py::list payloads;
	for (auto &payload_column : payload_columns) {
		payloads.append(payload_column);
	}
	result["payload_columns"] = payloads;

	py::list groups;
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
