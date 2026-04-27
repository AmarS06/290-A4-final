#include "dataframelib/eager_dataframe.hpp"

namespace dataframelib {

arrow::Result<EagerDataFrame> EagerDataFrame::from_arrow_table(const std::shared_ptr<arrow::Table>& table) {
    return arrow::Status::NotImplemented("Skeleton");
}

EagerDataFrame EagerDataFrame::from_columns(const ColumnMap& columns) {
    return EagerDataFrame({}, nullptr, 0);
}

EagerDataFrame EagerDataFrame::from_ordered_columns(std::vector<Column> columns) {
    return EagerDataFrame({}, nullptr, 0);
}

EagerDataFrame EagerDataFrame::read_csv(const std::string& path) {
    return EagerDataFrame({}, nullptr, 0);
}

EagerDataFrame EagerDataFrame::read_parquet(const std::string& path) {
    return EagerDataFrame({}, nullptr, 0);
}

arrow::Status EagerDataFrame::write_csv(const std::string& path) const {
    return arrow::Status::OK();
}

arrow::Status EagerDataFrame::write_parquet(const std::string& path) const {
    return arrow::Status::OK();
}

EagerDataFrame EagerDataFrame::select(const std::vector<std::string>& columns) const {
    return *this;
}

EagerDataFrame EagerDataFrame::select(const std::vector<Expr>& expressions) const {
    return *this;
}

EagerDataFrame EagerDataFrame::filter(const Expr& predicate) const {
    return *this;
}

EagerDataFrame EagerDataFrame::with_column(const std::string& name, const Expr& expr) const {
    return *this;
}

EagerDataFrame EagerDataFrame::head(int64_t n) const {
    return *this;
}

EagerDataFrame EagerDataFrame::sort(const std::vector<std::string>& columns,
                                    const std::vector<bool>& ascending) const {
    return *this;
}

GroupedEagerDataFrame EagerDataFrame::group_by(const std::vector<std::string>& keys) const {
    return GroupedEagerDataFrame(*this, keys);
}

EagerDataFrame EagerDataFrame::join(const EagerDataFrame& other,
                                    const std::vector<std::string>& on,
                                    const std::string& how) const {
    return *this;
}

arrow::Result<const Column*> EagerDataFrame::column(const std::string& name) const {
    return arrow::Status::KeyError("Skeleton");
}

EagerDataFrame GroupedEagerDataFrame::aggregate(const std::vector<std::pair<std::string, std::string>>& aggs) const {
    return df_;
}

}  