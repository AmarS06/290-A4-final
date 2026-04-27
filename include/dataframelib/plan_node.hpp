#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "dataframelib/expr.hpp"
namespace dataframelib {
class PlanNode {
public:
    enum class Kind {
        ScanCsv,
        ScanParquet,
        SelectNames,
        SelectExprs,
        Filter,
        WithColumn,
        GroupByAggregate,
        Join,
        Sort,
        Head,
    };
    virtual Kind kind() const=0;
    virtual std::string ToString() const=0;
    //inputs: primary and secondary
    virtual const std::shared_ptr<PlanNode>& input() const=0;
    virtual const std::shared_ptr<PlanNode>& right_input() const {
        static const std::shared_ptr<PlanNode> kNull;
        return kNull;
    }
    virtual ~PlanNode()=default;
};


class ScanCsvNode : public PlanNode {
public:
    explicit ScanCsvNode(std::string path) : path_(std::move(path)) {}
    Kind kind() const override { return Kind::ScanCsv; }
    std::string ToString() const override { return "scan_csv(\""+path_+"\")"; }
    const std::shared_ptr<PlanNode>& input() const override {
        static const std::shared_ptr<PlanNode> kNull;
        return kNull;
    }
    const std::string& path() const { return path_; }
private:
    std::string path_;
};

class ScanParquetNode : public PlanNode {
public:
    explicit ScanParquetNode(std::string path) : path_(std::move(path)) {}
    Kind kind() const override { return Kind::ScanParquet; }
    std::string ToString() const override { return "scan_parquet(\""+path_+"\")"; }
    const std::shared_ptr<PlanNode>& input() const override {
        static const std::shared_ptr<PlanNode> kNull;
        return kNull;
    }
    const std::string& path() const { return path_; }
private:
    std::string path_;
};

// select(column_names) — projects to a named subset of columns.
class SelectNamesNode : public PlanNode {
public:
    SelectNamesNode(std::shared_ptr<PlanNode> input, std::vector<std::string> columns)
        : input_(std::move(input)), columns_(std::move(columns)) {}
    Kind kind() const override { return Kind::SelectNames; }
    std::string ToString() const override;
    const std::shared_ptr<PlanNode>& input() const override { return input_; }
    const std::vector<std::string>& columns() const { return columns_; }
private:
    std::shared_ptr<PlanNode> input_;
    std::vector<std::string> columns_;
};

// select(expressions) — projects to a list of expressions (may include aliases).
class SelectExprsNode : public PlanNode {
public:
    SelectExprsNode(std::shared_ptr<PlanNode> input, std::vector<Expr> expressions)
        : input_(std::move(input)), expressions_(std::move(expressions)) {}
    Kind kind() const override { return Kind::SelectExprs; }
    std::string ToString() const override;
    const std::shared_ptr<PlanNode>& input() const override { return input_; }
    const std::vector<Expr>& expressions() const { return expressions_; }
private:
    std::shared_ptr<PlanNode> input_;
    std::vector<Expr> expressions_;
};

// filter(predicate) — keeps rows where predicate evaluates to true
// (null treated as false, matching eager semantics).
class FilterNode : public PlanNode {
public:
    FilterNode(std::shared_ptr<PlanNode> input, Expr predicate)
        : input_(std::move(input)), predicate_(std::move(predicate)) {}
    Kind kind() const override { return Kind::Filter; }
    std::string ToString() const override {
        return "filter("+predicate_.ToString()+")";
    }
    const std::shared_ptr<PlanNode>& input() const override { return input_; }
    const Expr& predicate() const { return predicate_; }
private:
    std::shared_ptr<PlanNode> input_;
    Expr predicate_;
};

// with_column(name, expr) — appends or replaces a column.
class WithColumnNode : public PlanNode {
public:
    WithColumnNode(std::shared_ptr<PlanNode> input, std::string name, Expr expr)
        : input_(std::move(input)), name_(std::move(name)), expr_(std::move(expr)) {}
    Kind kind() const override { return Kind::WithColumn; }
    std::string ToString() const override {
        return "with_column(\""+name_+"\", "+expr_.ToString()+")";
    }
    const std::shared_ptr<PlanNode>& input() const override { return input_; }
    const std::string& name() const { return name_; }
    const Expr& expr() const { return expr_; }
private:
    std::shared_ptr<PlanNode> input_;
    std::string name_;
    Expr expr_;
};

// group_by + aggregate — stored as one combined node because aggregate must
// always follow group_by; separating them would allow invalid plans.
class GroupByAggregateNode : public PlanNode {
public:
    using AggMap=std::unordered_map<std::string, Expr>;
    GroupByAggregateNode(std::shared_ptr<PlanNode> input,
                         std::vector<std::string> keys,
                         AggMap agg_map)
        : input_(std::move(input)),
          keys_(std::move(keys)),
          agg_map_(std::move(agg_map)) {}
    Kind kind() const override { return Kind::GroupByAggregate; }
    std::string ToString() const override;
    const std::shared_ptr<PlanNode>& input() const override { return input_; }
    const std::vector<std::string>& keys() const { return keys_; }
    const AggMap& agg_map() const { return agg_map_; }
private:
    std::shared_ptr<PlanNode> input_;
    std::vector<std::string> keys_;
    AggMap agg_map_;
};

// join(right, on, how) — binary node; left input is input_, right is right_.
class JoinNode : public PlanNode {
public:
    JoinNode(std::shared_ptr<PlanNode> left,
             std::shared_ptr<PlanNode> right,
             std::vector<std::string> on,
             std::string how)
        : left_(std::move(left)),
          right_(std::move(right)),
          on_(std::move(on)),
          how_(std::move(how)) {}
    Kind kind() const override { return Kind::Join; }
    std::string ToString() const override;
    const std::shared_ptr<PlanNode>& input() const override { return left_; }
    const std::shared_ptr<PlanNode>& right_input() const override { return right_; }
    const std::vector<std::string>& on() const { return on_; }
    const std::string& how() const { return how_; }
private:
    std::shared_ptr<PlanNode> left_;
    std::shared_ptr<PlanNode> right_;
    std::vector<std::string> on_;
    std::string how_;
};

// sort(columns, ascending)
class SortNode : public PlanNode {
public:
    SortNode(std::shared_ptr<PlanNode> input,
             std::vector<std::string> columns,
             std::vector<bool> ascending)
        : input_(std::move(input)),
          columns_(std::move(columns)),
          ascending_(std::move(ascending)) {}
    Kind kind() const override { return Kind::Sort; }
    std::string ToString() const override;
    const std::shared_ptr<PlanNode>& input() const override { return input_; }
    const std::vector<std::string>& columns() const { return columns_; }
    const std::vector<bool>& ascending() const { return ascending_; }
private:
    std::shared_ptr<PlanNode> input_;
    std::vector<std::string> columns_;
    std::vector<bool> ascending_;
};

// head(n) — limits output to first n rows.
class HeadNode : public PlanNode {
public:
    HeadNode(std::shared_ptr<PlanNode> input, int64_t n)
        : input_(std::move(input)), n_(n) {}
    Kind kind() const override { return Kind::Head; }
    std::string ToString() const override {
        return "head("+std::to_string(n_)+")";
    }
    const std::shared_ptr<PlanNode>& input() const override { return input_; }
    int64_t n() const { return n_; }
private:
    std::shared_ptr<PlanNode> input_;
    int64_t n_;
};

// Expose BuildDotGraph for DAG visualization (used in tests and explain)
void BuildDotGraph(const std::shared_ptr<PlanNode>& root, std::string* dot_graph_out);
} 
