#include "dataframelib/lazy_dataframe.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <arrow/result.h>
#include <arrow/status.h>
#include "dataframelib/optimizer.hpp"
#include "dataframelib/plan_node.hpp"

namespace dataframelib {

namespace {

std::string ShellQuote(const std::string& value) {
    std::string quoted = "'";
    for (char c : value) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted.push_back(c);
        }
    }
    quoted.push_back('\'');
    return quoted;
}

arrow::Result<EagerDataFrame> ExecutePlan(const std::shared_ptr<PlanNode>& root) {
    if (!root) {
        return arrow::Status::Invalid("collect: encountered null plan node.");
    }

    std::vector<std::shared_ptr<PlanNode>> stack = {root};
    std::vector<std::shared_ptr<PlanNode>> execution_order;
    std::unordered_set<const PlanNode*> processed;

    while (!stack.empty()) {
        auto curr = stack.back();
        bool all_inputs_processed = true;
        if (curr->right_input() && processed.find(curr->right_input().get()) == processed.end()) {
            stack.push_back(curr->right_input());
            all_inputs_processed = false;
        }
    
        if (curr->input() && processed.find(curr->input().get()) == processed.end()) {
            stack.push_back(curr->input());
            all_inputs_processed = false;
        }
        if (all_inputs_processed) {
            stack.pop_back();
            if (processed.find(curr.get()) == processed.end()) {
                execution_order.push_back(curr);
                processed.insert(curr.get());
            }
        }
    }
    std::unordered_map<const PlanNode*, EagerDataFrame> results;

    for (const auto& node : execution_order) {
        switch (node->kind()) {
            case PlanNode::Kind::ScanCsv: {
                const auto* scan = static_cast<const ScanCsvNode*>(node.get());
                auto df = EagerDataFrame::read_csv(scan->path());
                results.emplace(node.get(), std::move(df));
                break;
            }
            case PlanNode::Kind::ScanParquet: {
                const auto* scan = static_cast<const ScanParquetNode*>(node.get());
                auto df = EagerDataFrame::read_parquet(scan->path());
                results.emplace(node.get(), std::move(df));
                break;
            }
            case PlanNode::Kind::SelectNames: {
                const auto* select = static_cast<const SelectNamesNode*>(node.get());
                const auto& input_df = results.at(select->input().get());
                auto df = input_df.select(select->columns());
                results.emplace(node.get(), std::move(df));
                break;
            }
            case PlanNode::Kind::SelectExprs: {
                const auto* select = static_cast<const SelectExprsNode*>(node.get());
                const auto& input_df = results.at(select->input().get());
                auto df = input_df.select(select->expressions());
                results.emplace(node.get(), std::move(df));
                break;
            }
            case PlanNode::Kind::Filter: {
                const auto* filter = static_cast<const FilterNode*>(node.get());
                const auto& input_df = results.at(filter->input().get());
                auto df = input_df.filter(filter->predicate());
                results.emplace(node.get(), std::move(df));
                break;
            }
            case PlanNode::Kind::WithColumn: {
                const auto* with_col = static_cast<const WithColumnNode*>(node.get());
                const auto& input_df = results.at(with_col->input().get());
                auto df = input_df.with_column(with_col->name(), with_col->expr());
                results.emplace(node.get(), std::move(df));
                break;
            }
            case PlanNode::Kind::GroupByAggregate: {
                const auto* grouped = static_cast<const GroupByAggregateNode*>(node.get());
                const auto& input_df = results.at(grouped->input().get());
                
                std::vector<std::pair<std::string, std::string>> legacy_aggs;
                for (const auto& kv : grouped->agg_map()) {
                    std::string op_str = "unknown";
                    if (kv.second.node().kind() == ExprNode::Kind::Agg) {
                        const auto& agg_node = static_cast<const AggNode&>(kv.second.node());
                        switch (agg_node.op()) {
                            case AggOpKind::kSum: op_str = "sum"; break;
                            case AggOpKind::kMean: op_str = "mean"; break;
                            case AggOpKind::kCount: op_str = "count"; break;
                            case AggOpKind::kMin: op_str = "min"; break;
                            case AggOpKind::kMax: op_str = "max"; break;
                        }
                        
                        // Extract base col name from inner node
                        std::string base_col;
                        if (agg_node.input()->kind() == ExprNode::Kind::Col) {
                            base_col = static_cast<const ColNode&>(*agg_node.input()).name();
                        }
                        legacy_aggs.push_back({base_col, op_str});
                    }
                }
                
                auto df = input_df.group_by(grouped->keys()).aggregate(legacy_aggs);
                results.emplace(node.get(), std::move(df));
                break;
            }
            case PlanNode::Kind::Join: {
                const auto* join = static_cast<const JoinNode*>(node.get());
                const auto& left_df = results.at(join->input().get());
                const auto& right_df = results.at(join->right_input().get());
                auto df = left_df.join(right_df, join->on(), join->how());
                results.emplace(node.get(), std::move(df));
                break;
            }
            case PlanNode::Kind::Sort: {
                const auto* sort = static_cast<const SortNode*>(node.get());
                const auto& input_df = results.at(sort->input().get());
                auto df = input_df.sort(sort->columns(), sort->ascending());
                results.emplace(node.get(), std::move(df));
                break;
            }
            case PlanNode::Kind::Head: {
                const auto* head = static_cast<const HeadNode*>(node.get());
                const auto& input_df = results.at(head->input().get());
                auto df = input_df.head(head->n());
                results.emplace(node.get(), std::move(df));
                break;
            }
            default:
                return arrow::Status::NotImplemented("collect: unsupported plan node kind.");
        }
    }
    return results.at(root.get());
}

} 

std::string SelectNamesNode::ToString() const {
    std::string s = "select([";
    for (std::size_t i = 0; i < columns_.size(); ++i) {
        if (i > 0) s += ", ";
        s += "\"" + columns_[i] + "\"";
    }
    return s + "])";
}

std::string SelectExprsNode::ToString() const {
    std::string s = "select([";
    for (std::size_t i = 0; i < expressions_.size(); ++i) {
        if (i > 0) s += ", ";
        s += expressions_[i].ToString();
    }
    return s + "])";
}

std::string GroupByAggregateNode::ToString() const {
    std::string s = "group_by([";
    for (std::size_t i = 0; i < keys_.size(); ++i) {
        if (i > 0) s += ", ";
        s += "\"" + keys_[i] + "\"";
    }
    s += "]).aggregate({";
    bool first = true;
    for (const auto& [name, expr] : agg_map_) {
        if (!first) s += ", ";
        s += "\"" + name + "\": " + expr.ToString();
        first = false;
    }
    return s + "})";
}

std::string JoinNode::ToString() const {
    std::string s = "join(how=\"" + how_ + "\", on=[";
    for (std::size_t i = 0; i < on_.size(); ++i) {
        if (i > 0) s += ", ";
        s += "\"" + on_[i] + "\"";
    }
    return s + "])";
}

std::string SortNode::ToString() const {
    std::string s = "sort([";
    for (std::size_t i = 0; i < columns_.size(); ++i) {
        if (i > 0) s += ", ";
        s += "\"" + columns_[i] + "\" " + (ascending_[i] ? "asc" : "desc");
    }
    return s + "])";
}

LazyDataFrame::LazyDataFrame(std::shared_ptr<PlanNode> plan)
    : plan_(std::move(plan)) {}

LazyDataFrame LazyDataFrame::scan_csv(const std::string& path) {
    if (path.empty()) {
        throw std::invalid_argument("scan_csv: path must not be empty.");
    }
    return LazyDataFrame(std::make_shared<ScanCsvNode>(path));
}

LazyDataFrame LazyDataFrame::scan_parquet(const std::string& path) {
    if (path.empty()) {
        throw std::invalid_argument("scan_parquet: path must not be empty.");
    }
    return LazyDataFrame(std::make_shared<ScanParquetNode>(path));
}

LazyDataFrame LazyDataFrame::select(const std::vector<std::string>& columns) const {
    if (columns.empty()) {
        throw std::invalid_argument("select: column list must not be empty.");
    }
    return LazyDataFrame(std::make_shared<SelectNamesNode>(plan_, columns));
}

LazyDataFrame LazyDataFrame::select(const std::vector<Expr>& expressions) const {
    if (expressions.empty()) {
        throw std::invalid_argument("select: expression list must not be empty.");
    }
    return LazyDataFrame(std::make_shared<SelectExprsNode>(plan_, expressions));
}

LazyDataFrame LazyDataFrame::filter(const Expr& predicate) const {
    return LazyDataFrame(std::make_shared<FilterNode>(plan_, predicate));
}

LazyDataFrame LazyDataFrame::with_column(const std::string& name,
                                          const Expr& expr) const {
    if (name.empty()) {
        throw std::invalid_argument("with_column: column name must not be empty.");
    }
    return LazyDataFrame(std::make_shared<WithColumnNode>(plan_, name, expr));
}

GroupedLazyDataFrame LazyDataFrame::group_by(
        const std::vector<std::string>& keys) const {
    if (keys.empty()) {
        throw std::invalid_argument("group_by: key list must not be empty.");
    }
    return GroupedLazyDataFrame(plan_, keys);
}

LazyDataFrame LazyDataFrame::join(const LazyDataFrame& other,
                                   const std::vector<std::string>& on,
                                   const std::string& how) const {
    if (on.empty()) {
        throw std::invalid_argument("join: 'on' must specify at least one key column.");
    }
    if (how != "inner" && how != "left" && how != "right" && how != "outer") {
        throw std::invalid_argument(
            "join: 'how' must be one of 'inner', 'left', 'right', 'outer'. Got: " +
            how);
    }
    return LazyDataFrame(
        std::make_shared<JoinNode>(plan_, other.plan_, on, how));
}

LazyDataFrame LazyDataFrame::sort(const std::vector<std::string>& columns,
                                   const std::vector<bool>& ascending) const {
    if (columns.empty()) {
        throw std::invalid_argument("sort: column list must not be empty.");
    }
    if (columns.size() != ascending.size()) {
        throw std::invalid_argument(
            "sort: columns and ascending vectors must have the same length.");
    }
    return LazyDataFrame(std::make_shared<SortNode>(plan_, columns, ascending));
}

LazyDataFrame LazyDataFrame::head(int64_t n) const {
    if (n < 0) {
        throw std::invalid_argument("head: n must be >= 0.");
    }
    return LazyDataFrame(std::make_shared<HeadNode>(plan_, n));
}

EagerDataFrame LazyDataFrame::collect() const {
    QueryOptimizer optimizer;
    auto optimized_plan = optimizer.optimize(plan_);
    return ExecutePlan(optimized_plan).ValueOrDie();
}

arrow::Status LazyDataFrame::sink_csv(const std::string& path) const {
    return collect().write_csv(path);
}

arrow::Status LazyDataFrame::sink_parquet(const std::string& path) const {
    return collect().write_parquet(path);
}

arrow::Status LazyDataFrame::explain(const std::string& path) const {
    if (path.empty()) {
        return arrow::Status::Invalid("explain: output path must not be empty.");
    }
    if (!plan_) {
        return arrow::Status::Invalid("explain: plan is null.");
    }

    std::error_code fs_error;
    const std::filesystem::path output_path(path);
    const auto parent = output_path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, fs_error);
        if (fs_error) {
            return arrow::Status::IOError(
                "explain: failed to create output directory '", parent.string(), "': ", fs_error.message());
        }
    }

    // Reliable helper: writes the .dot text file, asks the terminal to convert it to PNG, then deletes the .dot
    auto render_graph = [](const std::string& dot_content, const std::string& out_png) -> arrow::Status {
        const std::string dot_path = out_png + ".dot";
        {
            std::ofstream dot_file(dot_path, std::ios::out | std::ios::trunc);
            if (!dot_file.is_open()) return arrow::Status::IOError("Failed to open dot file.");
            dot_file << dot_content;
        }

        const std::string command = "dot -Tpng " + ShellQuote(dot_path) + " -o " + ShellQuote(out_png);
        const int rc = std::system(command.c_str());

        std::error_code ec;
        std::filesystem::remove(dot_path, ec); // Clean up the temp .dot file

        if (rc != 0) {
            return arrow::Status::IOError("explain: Graphviz 'dot' command failed.");
        }
        return arrow::Status::OK();
    };

    std::string base_path = path;
    std::string extension = "";
    size_t dot_pos = path.find_last_of('.');
    if (dot_pos != std::string::npos && dot_pos > path.find_last_of('/')) {
        base_path = path.substr(0, dot_pos);
        extension = path.substr(dot_pos);
    }

    const std::string logical_path = base_path + "_logical" + extension;
    const std::string optimized_path = base_path + "_optimized" + extension;
    std::string logical_dot;
    BuildDotGraph(plan_, &logical_dot);
    ARROW_RETURN_NOT_OK(render_graph(logical_dot, logical_path));
    QueryOptimizer optimizer;
    auto optimized_plan = optimizer.optimize(plan_);
    std::string optimized_dot;
    BuildDotGraph(optimized_plan, &optimized_dot);
    ARROW_RETURN_NOT_OK(render_graph(optimized_dot, optimized_path));

    return arrow::Status::OK();
}

GroupedLazyDataFrame::GroupedLazyDataFrame(std::shared_ptr<PlanNode> input,
                                            std::vector<std::string> keys)
    : input_(std::move(input)), keys_(std::move(keys)) {}

LazyDataFrame GroupedLazyDataFrame::aggregate(const std::vector<std::pair<std::string, std::string>>& aggs) const {
    if (aggs.empty()) {
        throw std::invalid_argument("aggregate: aggregation map must not be empty.");
    }
    std::unordered_map<std::string, Expr> internal_agg_map;
    for (const auto& pair : aggs) {
        std::string output_name = pair.first + "_" + pair.second;
        Expr expr = col(pair.first);
        if (pair.second == "sum") expr = expr.sum();
        else if (pair.second == "mean") expr = expr.mean();
        else if (pair.second == "count") expr = expr.count();
        else if (pair.second == "min") expr = expr.min();
        else if (pair.second == "max") expr = expr.max();
        internal_agg_map.emplace(output_name, expr);
    }
    
    return LazyDataFrame(
        std::make_shared<GroupByAggregateNode>(input_, keys_, internal_agg_map));
}

} 