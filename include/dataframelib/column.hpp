#pragma once

#include <memory>
#include <string>

#include <arrow/api.h>

#include "dataframelib/types.hpp"

namespace dataframelib {

enum class BinaryOp {
    kAdd,
    kSubtract,
    kMultiply,
    kDivide,
    kModulo,
};

enum class ComparisonOp {
    kEqual,
    kNotEqual,
    kLess,
    kLessEqual,
    kGreater,
    kGreaterEqual,
};

enum class BooleanBinaryOp {
    kAnd,
    kOr,
};

enum class StringUnaryOp {
    kLength,
    kToLower,
    kToUpper,
};

enum class StringPredicateOp {
    kContains,
    kStartsWith,
    kEndsWith,
};

enum class NullPredicateOp {
    kIsNull,
    kIsNotNull,
};

class Column {
public:
    static arrow::Result<Column> Make(std::string name, std::shared_ptr<arrow::Array> data);

    const std::string& name() const { return name_; }
    const std::shared_ptr<arrow::Array>& data() const { return data_; }
    DType dtype() const { return dtype_; }
    int64_t length() const { return data_->length(); }

private:
    Column(std::string name, std::shared_ptr<arrow::Array> data, DType dtype)
        : name_(std::move(name)), data_(std::move(data)), dtype_(dtype) {}

    std::string name_;
    std::shared_ptr<arrow::Array> data_;
    DType dtype_;
};

arrow::Result<Column> ApplyBinaryNumericOp(
    const Column& lhs,
    const Column& rhs,
    BinaryOp op,
    std::string output_name);

arrow::Result<Column> ApplyUnaryAbsOp(
    const Column& input,
    std::string output_name);

arrow::Result<Column> ApplyComparisonOp(
    const Column& lhs,
    const Column& rhs,
    ComparisonOp op,
    std::string output_name);

arrow::Result<Column> ApplyBooleanBinaryOp(
    const Column& lhs,
    const Column& rhs,
    BooleanBinaryOp op,
    std::string output_name);

arrow::Result<Column> ApplyBooleanNotOp(
    const Column& input,
    std::string output_name);

arrow::Result<Column> ApplyNullPredicateOp(
    const Column& input,
    NullPredicateOp op,
    std::string output_name);

arrow::Result<Column> ApplyStringUnaryOp(
    const Column& input,
    StringUnaryOp op,
    std::string output_name);

arrow::Result<Column> ApplyStringPredicateOp(
    const Column& input,
    StringPredicateOp op,
    std::string pattern,
    std::string output_name);

} 
