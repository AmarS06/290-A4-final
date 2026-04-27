#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <initializer_list>
#include <arrow/result.h>
#include "dataframelib/eager_dataframe.hpp"
#include "dataframelib/expr.hpp"
#include "dataframelib/plan_node.hpp"
namespace dataframelib {
class GroupedLazyDataFrame;
class LazyDataFrame {
public:
    static LazyDataFrame scan_csv(const std::string& path);
    static LazyDataFrame scan_parquet(const std::string& path);
    LazyDataFrame select(const std::vector<std::string>& columns) const;
    LazyDataFrame select(const std::vector<Expr>& expressions) const;
    // OVERLOAD
    LazyDataFrame select(std::initializer_list<const char*> columns) const {
        std::vector<std::string> cols;
        for (auto c : columns) cols.push_back(c);
        return select(cols);
    }
    LazyDataFrame filter(const Expr& predicate) const;
    LazyDataFrame with_column(const std::string& name, const Expr& expr) const;
    GroupedLazyDataFrame group_by(const std::vector<std::string>& keys) const;
    // Declaration only; implemented at the bottom of the file
    GroupedLazyDataFrame group_by(std::initializer_list<const char*> keys) const;
    LazyDataFrame join(const LazyDataFrame& other,
                       const std::vector<std::string>& on,
                       const std::string& how) const;
    // OVERLOAD
    LazyDataFrame join(const LazyDataFrame& other,
                       std::initializer_list<const char*> on,
                       const std::string& how) const {
        std::vector<std::string> on_keys;
        for (auto k : on) on_keys.push_back(k);
        return join(other, on_keys, how);
    }
    LazyDataFrame sort(const std::vector<std::string>& columns,
                       const std::vector<bool>& ascending) const;
    // OVERLOAD
    LazyDataFrame sort(std::initializer_list<const char*> columns, bool ascending) const {
        std::vector<std::string> cols;
        for (auto c : columns) cols.push_back(c);
        return sort(cols, std::vector<bool>(cols.size(), ascending));
    }
    LazyDataFrame head(int64_t n) const;
    EagerDataFrame collect() const;
    arrow::Status sink_csv(const std::string& path) const;
    arrow::Status sink_parquet(const std::string& path) const;
    arrow::Status explain(const std::string& path) const;
    const std::shared_ptr<PlanNode>& plan() const { return plan_; }
private:
    explicit LazyDataFrame(std::shared_ptr<PlanNode> plan);
    std::shared_ptr<PlanNode> plan_;
    friend class GroupedLazyDataFrame;
};

class GroupedLazyDataFrame {
public:
    LazyDataFrame aggregate(const std::vector<std::pair<std::string, std::string>>& aggs) const;
    const std::vector<std::string>& keys() const { return keys_; }
private:
    GroupedLazyDataFrame(std::shared_ptr<PlanNode> input,
                         std::vector<std::string> keys);
    std::shared_ptr<PlanNode> input_;
    std::vector<std::string> keys_;
    friend class LazyDataFrame;
};

inline GroupedLazyDataFrame LazyDataFrame::group_by(std::initializer_list<const char*> keys) const {
    std::vector<std::string> ks;
    for (auto k : keys) ks.push_back(k);
    return group_by(ks);
}

} 