#include "dataframelib/column.hpp"

namespace dataframelib {

DType ToDType(const std::shared_ptr<arrow::DataType>& type) {
    return DType::kInt32; 
}

std::string ToString(DType dtype) {
    return "";
}

bool IsSupportedType(const std::shared_ptr<arrow::DataType>& type) {
    return false;
}

bool IsNumericType(const std::shared_ptr<arrow::DataType>& type) {
    return false;
}

bool IsFloatingType(const std::shared_ptr<arrow::DataType>& type) {
    return false;
}

arrow::Result<Column> Column::Make(std::string name, std::shared_ptr<arrow::Array> data) {
    return arrow::Status::NotImplemented("Skeleton");
}

arrow::Result<Column> ApplyBinaryNumericOp(const Column& lhs, const Column& rhs, BinaryOp op, std::string output_name) {
    return arrow::Status::NotImplemented("Skeleton");
}

arrow::Result<Column> ApplyUnaryAbsOp(const Column& input, std::string output_name) {
    return arrow::Status::NotImplemented("Skeleton");
}

arrow::Result<Column> ApplyComparisonOp(const Column& lhs, const Column& rhs, ComparisonOp op, std::string output_name) {
    return arrow::Status::NotImplemented("Skeleton");
}

arrow::Result<Column> ApplyBooleanBinaryOp(const Column& lhs, const Column& rhs, BooleanBinaryOp op, std::string output_name) {
    return arrow::Status::NotImplemented("Skeleton");
}

arrow::Result<Column> ApplyBooleanNotOp(const Column& input, std::string output_name) {
    return arrow::Status::NotImplemented("Skeleton");
}

arrow::Result<Column> ApplyNullPredicateOp(const Column& input, NullPredicateOp op, std::string output_name) {
    return arrow::Status::NotImplemented("Skeleton");
}

arrow::Result<Column> ApplyStringUnaryOp(const Column& input, StringUnaryOp op, std::string output_name) {
    return arrow::Status::NotImplemented("Skeleton");
}

arrow::Result<Column> ApplyStringPredicateOp(const Column& input, StringPredicateOp op, std::string pattern, std::string output_name) {
    return arrow::Status::NotImplemented("Skeleton");
}

} 