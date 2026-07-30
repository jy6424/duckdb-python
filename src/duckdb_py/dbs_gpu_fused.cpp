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

struct DirectPipelineRawBatch {
	string fact_path;
	std::shared_ptr<GroupMapping> mapping;
	unique_ptr<DataChunk> chunk;
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

static void ReadPipelineRawBatches(DuckDB &db, const vector<string> &fact_paths, const string &payload_column,
                                   const string &join_key, const string &group_column, const string &dimension_file,
                                   BlockingQueue<PipelineRawBatch> &raw_queue, std::exception_ptr &error_out,
                                   std::mutex &error_lock) {
	try {
		Connection connection(db);
		for (auto &fact_path : fact_paths) {
			auto dimension_path = ParentPath(fact_path) + "/" + dimension_file;
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

static void ReadDirectPipelineRawBatches(DuckDB &db, const vector<string> &fact_paths, const string &payload_column,
                                         const string &join_key, const string &group_column,
                                         const string &dimension_file,
                                         BlockingQueue<DirectPipelineRawBatch> &raw_queue,
                                         std::exception_ptr &error_out, std::mutex &error_lock) {
	try {
		Connection connection(db);
		for (auto &fact_path : fact_paths) {
			auto dimension_path = ParentPath(fact_path) + "/" + dimension_file;
			auto mapping =
			    std::make_shared<GroupMapping>(ReadGroupMapping(connection, dimension_path, join_key, group_column));
			auto result = RunStreamingQuery(connection, BuildProbeQuery(fact_path, join_key, payload_column));

			while (true) {
				auto chunk = result->Fetch();
				if (!chunk || chunk->size() == 0) {
					break;
				}

				DirectPipelineRawBatch batch;
				batch.fact_path = fact_path;
				batch.mapping = mapping;
				batch.chunk = std::move(chunk);
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

static void StartDirectPipelineBuffer(FusedLatAggPipelineFuncs pipeline, void *handle,
                                      BlockingQueue<idx_t> &free_slots, idx_t target_batch_rows,
                                      const DirectPipelineRawBatch &raw, DirectPipelineBuffer &current) {
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

static idx_t AppendDirectPipelineRows(DataChunk &chunk, idx_t &row_offset, DirectPipelineBuffer &current,
                                      idx_t target_batch_rows) {
	auto count = chunk.size();
	auto join_data = FlatVector::GetData<int64_t>(chunk.data[0]);
	auto value_data = FlatVector::GetData<double>(chunk.data[1]);
	auto valid_data = FlatVector::GetData<uint8_t>(chunk.data[2]);
	auto &join_validity = FlatVector::Validity(chunk.data[0]);
	idx_t appended = 0;

	while (row_offset < count && current.count < target_batch_rows) {
		if (join_validity.RowIsValid(row_offset)) {
			current.join_keys[current.count] = join_data[row_offset];
			current.values[current.count] = value_data[row_offset];
			current.validity[current.count] = valid_data[row_offset];
			current.count++;
			appended++;
		}
		row_offset++;
	}
	return appended;
}

static void PrepareDirectPipelineInputBatches(FusedLatAggPipelineFuncs pipeline, void *handle,
                                             BlockingQueue<DirectPipelineRawBatch> &raw_queue,
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

		DirectPipelineRawBatch raw;
		while (raw_queue.Pop(raw)) {
			idx_t row_offset = 0;
			bool counted_chunk = false;
			while (raw.chunk && row_offset < raw.chunk->size()) {
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

				AppendDirectPipelineRows(*raw.chunk, row_offset, current, target_batch_rows);
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
	const bool pipeline_direct_mode = mode == "pipeline-mapped" || mode == "pipeline-managed";
	const bool pipeline_mode = mode == "pipeline-device" || pipeline_direct_mode;
	const bool mapped = mode == "mapped" || mode == "pipeline-mapped";
	const int pipeline_memory_mode = mode == "pipeline-mapped" ? 1 : (mode == "pipeline-managed" ? 2 : 0);
	if (!mapped && mode != "device" && mode != "pipeline-managed" && !pipeline_mode) {
		throw InvalidInputException(
		    "mode must be 'device', 'mapped', 'pipeline-device', 'pipeline-mapped', or 'pipeline-managed'");
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
			BlockingQueue<DirectPipelineRawBatch> raw_queue(8);
			BlockingQueue<DirectPipelineInputBatch> input_queue(PIPELINE_SLOTS);
			BlockingQueue<idx_t> free_slots(0);
			for (idx_t slot = 0; slot < PIPELINE_SLOTS; slot++) {
				free_slots.Push(slot);
			}

			std::thread reader_thread(ReadDirectPipelineRawBatches, std::ref(db), std::cref(fact_paths),
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
	} else {
		auto fused_agg = LoadFusedLatAgg(lib_path, mapped);

		for (auto &fact_path : fact_paths) {
			auto dimension_path = ParentPath(fact_path) + "/" + dimension_file;
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

} // namespace

void RegisterDBSGPUFused(py::module_ &m) {
	m.def("dbs_gpu_fused_lat_pipeline", &DBSGPUFusedLatPipeline, py::arg("fact_paths"),
	      py::arg("payload_column") = "qicps", py::arg("join_key") = "grid", py::arg("group_column") = "lats",
	      py::arg("dimension_file") = "grid.parquet", py::arg("lib_path") = "",
	      py::arg("mode") = "pipeline-managed");
}

} // namespace duckdb
