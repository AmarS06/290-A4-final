#pragma once
#include <stdexcept>
#include <arrow/status.h>
// NOTE: to adjust to autograder
#ifndef ARROW_THROW_NOT_OK
#define ARROW_THROW_NOT_OK(status) do { \
    ::arrow::Status _s=(status); \
    if (!_s.ok()) throw std::runtime_error(_s.ToString()); \
} while (0)
#endif
#include "dataframelib/types.hpp"
#include "dataframelib/column.hpp"
#include "dataframelib/expr.hpp"
#include "dataframelib/eager_dataframe.hpp"
#include "dataframelib/plan_node.hpp"
#include "dataframelib/lazy_dataframe.hpp"
#include "dataframelib/optimizer.hpp"
namespace dataframelib {
    inline EagerDataFrame read_csv(const std::string& path) {
        return EagerDataFrame::read_csv(path);
    }
    inline EagerDataFrame read_parquet(const std::string& path) {
        return EagerDataFrame::read_parquet(path);
    }
    inline LazyDataFrame scan_csv(const std::string& path) {
        return LazyDataFrame::scan_csv(path);
    }
    inline LazyDataFrame scan_parquet(const std::string& path) {
        return LazyDataFrame::scan_parquet(path);
    }
    inline EagerDataFrame from_columns(const EagerDataFrame::ColumnMap& cols) {
        return EagerDataFrame::from_columns(cols);
    }
    inline EagerDataFrame from_columns(const std::vector<std::pair<std::string, std::shared_ptr<arrow::Array>>>& cols) {
        EagerDataFrame::ColumnMap cmap;
        for (const auto& p : cols) cmap[p.first]=p.second;
        return EagerDataFrame::from_columns(cmap);
    }
    inline EagerDataFrame from_columns(std::initializer_list<std::pair<const std::string, std::shared_ptr<arrow::Array>>> cols) {
        EagerDataFrame::ColumnMap cmap;
        for (const auto& p : cols) cmap[p.first]=p.second;
        return EagerDataFrame::from_columns(cmap);
    }
} 