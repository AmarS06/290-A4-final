#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <initializer_list>

#include <arrow/api.h>

#include "dataframelib/column.hpp"
#include "dataframelib/expr.hpp"

namespace dataframelib {

class GroupedEagerDataFrame;

class EagerDataFrame {
public:
    using ColumnMap = std::unordered_map<std::string, std::shared_ptr<arrow::Array>>;

    static EagerDataFrame from_columns(const ColumnMap& columns);
    static EagerDataFrame from_ordered_columns(std::vector<Column> columns);
    static EagerDataFrame read_csv(const std::string& path);
    static EagerDataFrame read_parquet(const std::string& path);
    
    arrow::Status write_csv(const std::string& path) const;
    arrow::Status write_parquet(const std::string& path) const;

    EagerDataFrame select(const std::vector<std::string>& columns) const;
    EagerDataFrame select(const std::vector<Expr>& expressions) const;
    
    // OVERLOAD
    EagerDataFrame select(std::initializer_list<const char*> columns) const {
        std::vector<std::string> cols;
        for (auto c : columns) cols.push_back(c);
        return select(cols);
    }

    EagerDataFrame filter(const Expr& predicate) const;
    EagerDataFrame with_column(const std::string& name, const Expr& expr) const;
    EagerDataFrame head(int64_t n) const;
    
    EagerDataFrame sort(const std::vector<std::string>& columns,
                        const std::vector<bool>& ascending) const;
                        
    // OVERLOAD
    EagerDataFrame sort(std::initializer_list<const char*> columns, bool ascending) const {
        std::vector<std::string> cols;
        for (auto c : columns) cols.push_back(c);
        return sort(cols, std::vector<bool>(cols.size(), ascending));
    }

    GroupedEagerDataFrame group_by(const std::vector<std::string>& keys) const;

    // Declaration only; implemented at the bottom of the file
    GroupedEagerDataFrame group_by(std::initializer_list<const char*> keys) const;

    EagerDataFrame join(const EagerDataFrame& other,
                        const std::vector<std::string>& on,
                        const std::string& how) const;

    // OVERLOAD
    EagerDataFrame join(const EagerDataFrame& other,
                        std::initializer_list<const char*> on,
                        const std::string& how) const {
        std::vector<std::string> on_keys;
        for (auto k : on) on_keys.push_back(k);
        return join(other, on_keys, how);
    }

    int64_t num_rows() const { return num_rows_; }
    int64_t num_columns() const { return static_cast<int64_t>(columns_.size()); }

    const std::vector<Column>& columns() const { return columns_; }
    const std::shared_ptr<arrow::Schema>& schema() const { return schema_; }

    arrow::Result<const Column*> column(const std::string& name) const;

private:
    EagerDataFrame(std::vector<Column> columns, std::shared_ptr<arrow::Schema> schema, int64_t num_rows)
        : columns_(std::move(columns)), schema_(std::move(schema)), num_rows_(num_rows) {}

    static arrow::Result<EagerDataFrame> from_arrow_table(const std::shared_ptr<arrow::Table>& table);

    std::vector<Column> columns_;
    std::shared_ptr<arrow::Schema> schema_;
    int64_t num_rows_;
};

class GroupedEagerDataFrame {
public:
    EagerDataFrame aggregate(const std::vector<std::pair<std::string, std::string>>& aggs) const;

private:
    EagerDataFrame df_;
    std::vector<std::string> keys_;

    GroupedEagerDataFrame(EagerDataFrame df, std::vector<std::string> keys)
        : df_(std::move(df)), keys_(std::move(keys)) {}

    friend class EagerDataFrame;
};

inline GroupedEagerDataFrame EagerDataFrame::group_by(std::initializer_list<const char*> keys) const {
    std::vector<std::string> ks;
    for (auto k : keys) ks.push_back(k);
    return group_by(ks);
}

}  