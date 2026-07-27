#include "duckdb_python/pybind11/pybind_wrapper.hpp"

#include "duckdb.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/vector.hpp"

#include <chrono>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <map>

namespace py = pybind11;

namespace duckdb {

namespace {

using FusedLatAggFunc = int (*)(const int64_t *grids, const double *values, const uint8_t *value_validity,
                                uint64_t count, int64_t grid_min, int64_t grid_max, const int32_t *grid_to_group,
                                uint64_t build_size, uint64_t group_count, double *sum_out, uint64_t *count_out,
                                uint64_t *row_count_out);

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

static ProbeColumns ReadProbeColumns(Connection &connection, const string &fact_path, const string &join_key,
                                     const string &payload_column) {
	auto query = StringUtil::Format(
	    "SELECT %s::BIGINT AS join_key, COALESCE(%s, 0)::DOUBLE AS value, "
	    "CASE WHEN %s IS NULL THEN 0 ELSE 1 END::UTINYINT AS value_valid FROM read_parquet('%s')",
	    QuoteIdentifier(join_key), QuoteIdentifier(payload_column), QuoteIdentifier(payload_column),
	    EscapeSQLString(fact_path));
	auto result = RunStreamingQuery(connection, query);
	ProbeColumns columns;

	while (true) {
		auto chunk = result->Fetch();
		if (!chunk || chunk->size() == 0) {
			break;
		}
		auto count = chunk->size();
		auto join_data = FlatVector::GetData<int64_t>(chunk->data[0]);
		auto value_data = FlatVector::GetData<double>(chunk->data[1]);
		auto valid_data = FlatVector::GetData<uint8_t>(chunk->data[2]);
		auto &join_validity = FlatVector::Validity(chunk->data[0]);
		for (idx_t row = 0; row < count; row++) {
			if (!join_validity.RowIsValid(row)) {
				continue;
			}
			columns.join_keys.push_back(join_data[row]);
			columns.values.push_back(value_data[row]);
			columns.validity.push_back(valid_data[row]);
		}
	}
	return columns;
}

static py::dict DBSGPUFusedLatPipeline(const py::iterable &fact_paths_p, const string &payload_column,
                                       const string &join_key, const string &group_column,
                                       const string &dimension_file, const string &lib_path, const string &mode) {
	const bool mapped = mode == "mapped";
	if (!mapped && mode != "device") {
		throw InvalidInputException("mode must be either 'device' or 'mapped'");
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
	auto fused_agg = LoadFusedLatAgg(lib_path, mapped);

	std::map<double, double> total_sum;
	std::map<double, uint64_t> total_count;
	uint64_t total_rows = 0;

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
	      py::arg("dimension_file") = "grid.parquet", py::arg("lib_path") = "", py::arg("mode") = "device");
}

} // namespace duckdb
