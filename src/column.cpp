#include "dataframelib/column.hpp"
#include <cctype>
#include <cmath>
#include <utility>
#include <vector>
#include <arrow/type_traits.h>
namespace dataframelib {
namespace {
arrow::Status RejectNaNInFloatingArrays(const std::shared_ptr<arrow::Array>& array) {
    if (!IsFloatingType(array->type())) {
        return arrow::Status::OK();
    }
    if (array->type_id()==arrow::Type::FLOAT) {
        const auto& float_array=static_cast<const arrow::FloatArray&>(*array);
        for (int64_t i=0; i<float_array.length(); ++i) {
            if (!float_array.IsNull(i)&&std::isnan(float_array.Value(i))) {
                return arrow::Status::Invalid("NaN is not allowed; represent missing values using null.");
            }
        }
        return arrow::Status::OK();
    }
    const auto& double_array=static_cast<const arrow::DoubleArray&>(*array);
    for (int64_t i=0; i<double_array.length(); ++i) {
        if (!double_array.IsNull(i)&&std::isnan(double_array.Value(i))) {
            return arrow::Status::Invalid("NaN is not allowed; represent missing values using null.");
        }
    }
    return arrow::Status::OK();
}

std::shared_ptr<arrow::DataType> ResolvePromotedNumericType(
    const std::shared_ptr<arrow::DataType>& left,
    const std::shared_ptr<arrow::DataType>& right) {
    const bool has_double=left->id()==arrow::Type::DOUBLE||right->id()==arrow::Type::DOUBLE;
    if (has_double) {
        return arrow::float64();
    }
    const bool has_float=left->id()==arrow::Type::FLOAT||right->id()==arrow::Type::FLOAT;
    if (has_float) {
        return arrow::float32();
    }
    const bool has_int64=left->id()==arrow::Type::INT64||right->id()==arrow::Type::INT64;
    if (has_int64) {
        return arrow::int64();
    }
    return arrow::int32();
}

double GetNumericAsDouble(const std::shared_ptr<arrow::Array>& array, int64_t index) {
    switch (array->type_id()) {
        case arrow::Type::INT32:
            return static_cast<double>(static_cast<const arrow::Int32Array&>(*array).Value(index));
        case arrow::Type::INT64:
            return static_cast<double>(static_cast<const arrow::Int64Array&>(*array).Value(index));
        case arrow::Type::FLOAT:
            return static_cast<double>(static_cast<const arrow::FloatArray&>(*array).Value(index));
        case arrow::Type::DOUBLE:
            return static_cast<const arrow::DoubleArray&>(*array).Value(index);
        default:
            return 0.0;
    }
}

arrow::Result<double> ApplyBinaryOpOnScalars(double lhs, double rhs, BinaryOp op) {
    switch (op) {
        case BinaryOp::kAdd:
            return lhs+rhs;
        case BinaryOp::kSubtract:
            return lhs-rhs;
        case BinaryOp::kMultiply:
            return lhs*rhs;
        case BinaryOp::kDivide:
            if (rhs==0.0) {
                return arrow::Status::Invalid("Division by zero is not allowed.");
            }
            return lhs/rhs;
        case BinaryOp::kModulo:
            if (rhs==0.0) {
                return arrow::Status::Invalid("Modulo by zero is not allowed.");
            }
            return std::fmod(lhs, rhs);
    }
    return arrow::Status::NotImplemented("Unsupported binary op.");
}

arrow::Result<std::shared_ptr<arrow::Array>> EvaluateNumericBinaryOp(
    const std::shared_ptr<arrow::Array>& lhs,
    const std::shared_ptr<arrow::Array>& rhs,
    const std::shared_ptr<arrow::DataType>& output_type,
    BinaryOp op) {
    if (op==BinaryOp::kModulo &&
        (output_type->id()==arrow::Type::INT32||output_type->id()==arrow::Type::INT64)) {
        auto get_i64=[](const std::shared_ptr<arrow::Array>& a, int64_t i) -> int64_t {
            return a->type_id()==arrow::Type::INT32
                ? static_cast<int64_t>(static_cast<const arrow::Int32Array&>(*a).Value(i))
                : static_cast<const arrow::Int64Array&>(*a).Value(i);
        };
        if (output_type->id()==arrow::Type::INT64) {
            arrow::Int64Builder b;
            for (int64_t i=0; i<lhs->length(); ++i) {
                if (lhs->IsNull(i)||rhs->IsNull(i)) { ARROW_RETURN_NOT_OK(b.AppendNull()); continue; }
                const int64_t r=get_i64(rhs, i);
                if (r==0) return arrow::Status::Invalid("Modulo by zero is not allowed.");
                ARROW_RETURN_NOT_OK(b.Append(get_i64(lhs, i)%r));
            }
            std::shared_ptr<arrow::Array> out;
            ARROW_RETURN_NOT_OK(b.Finish(&out));
            return out;
        }
        arrow::Int32Builder b;
        for (int64_t i=0; i<lhs->length(); ++i) {
            if (lhs->IsNull(i)||rhs->IsNull(i)) { ARROW_RETURN_NOT_OK(b.AppendNull()); continue; }
            const int64_t r=get_i64(rhs, i);
            if (r==0) return arrow::Status::Invalid("Modulo by zero is not allowed.");
            ARROW_RETURN_NOT_OK(b.Append(static_cast<int32_t>(get_i64(lhs, i)%r)));
        }
        std::shared_ptr<arrow::Array> out;
        ARROW_RETURN_NOT_OK(b.Finish(&out));
        return out;
    }
    if (output_type->id()==arrow::Type::FLOAT) {
        arrow::FloatBuilder builder;
        for (int64_t i=0; i<lhs->length(); ++i) {
            if (lhs->IsNull(i)||rhs->IsNull(i)) {
                ARROW_RETURN_NOT_OK(builder.AppendNull());
                continue;
            }
            ARROW_ASSIGN_OR_RAISE(double value, ApplyBinaryOpOnScalars(GetNumericAsDouble(lhs, i), GetNumericAsDouble(rhs, i), op));
            ARROW_RETURN_NOT_OK(builder.Append(static_cast<float>(value)));
        }
        std::shared_ptr<arrow::Array> out;
        ARROW_RETURN_NOT_OK(builder.Finish(&out));
        return out;
    }
    if (output_type->id()==arrow::Type::DOUBLE) {
        arrow::DoubleBuilder builder;
        for (int64_t i=0; i<lhs->length(); ++i) {
            if (lhs->IsNull(i)||rhs->IsNull(i)) {
                ARROW_RETURN_NOT_OK(builder.AppendNull());
                continue;
            }
            ARROW_ASSIGN_OR_RAISE(double value, ApplyBinaryOpOnScalars(GetNumericAsDouble(lhs, i), GetNumericAsDouble(rhs, i), op));
            ARROW_RETURN_NOT_OK(builder.Append(value));
        }
        std::shared_ptr<arrow::Array> out;
        ARROW_RETURN_NOT_OK(builder.Finish(&out));
        return out;
    }
    if (output_type->id()==arrow::Type::INT64) {
        arrow::Int64Builder builder;
        for (int64_t i=0; i<lhs->length(); ++i) {
            if (lhs->IsNull(i)||rhs->IsNull(i)) {
                ARROW_RETURN_NOT_OK(builder.AppendNull());
                continue;
            }
            ARROW_ASSIGN_OR_RAISE(double value, ApplyBinaryOpOnScalars(GetNumericAsDouble(lhs, i), GetNumericAsDouble(rhs, i), op));
            ARROW_RETURN_NOT_OK(builder.Append(static_cast<int64_t>(value)));
        }
        std::shared_ptr<arrow::Array> out;
        ARROW_RETURN_NOT_OK(builder.Finish(&out));
        return out;
    }
    arrow::Int32Builder builder;
    for (int64_t i=0; i<lhs->length(); ++i) {
        if (lhs->IsNull(i)||rhs->IsNull(i)) {
            ARROW_RETURN_NOT_OK(builder.AppendNull());
            continue;
        }
        ARROW_ASSIGN_OR_RAISE(double value, ApplyBinaryOpOnScalars(GetNumericAsDouble(lhs, i), GetNumericAsDouble(rhs, i), op));
        ARROW_RETURN_NOT_OK(builder.Append(static_cast<int32_t>(value)));
    }
    std::shared_ptr<arrow::Array> out;
    ARROW_RETURN_NOT_OK(builder.Finish(&out));
    return out;
}

}  // namespace

bool IsSupportedType(const std::shared_ptr<arrow::DataType>& type) {
    switch (type->id()) {
        case arrow::Type::INT32:
        case arrow::Type::INT64:
        case arrow::Type::FLOAT:
        case arrow::Type::DOUBLE:
        case arrow::Type::STRING:
        case arrow::Type::BOOL:
            return true;
        default:
            return false;
    }
}

bool IsNumericType(const std::shared_ptr<arrow::DataType>& type) {
    switch (type->id()) {
        case arrow::Type::INT32:
        case arrow::Type::INT64:
        case arrow::Type::FLOAT:
        case arrow::Type::DOUBLE:
            return true;
        default:
            return false;
    }
}

bool IsFloatingType(const std::shared_ptr<arrow::DataType>& type) {
    return type->id()==arrow::Type::FLOAT||type->id()==arrow::Type::DOUBLE;
}

DType ToDType(const std::shared_ptr<arrow::DataType>& type) {
    switch (type->id()) {
        case arrow::Type::INT32:
            return DType::kInt32;
        case arrow::Type::INT64:
            return DType::kInt64;
        case arrow::Type::FLOAT:
            return DType::kFloat32;
        case arrow::Type::DOUBLE:
            return DType::kFloat64;
        case arrow::Type::STRING:
            return DType::kString;
        case arrow::Type::BOOL:
            return DType::kBoolean;
        default:
            break;
    }
    throw std::invalid_argument("Unsupported Arrow type for COP290 DataFrameLib.");
}

std::string ToString(DType dtype) {
    switch (dtype) {
        case DType::kInt32:
            return "int32";
        case DType::kInt64:
            return "int64";
        case DType::kFloat32:
            return "float32";
        case DType::kFloat64:
            return "float64";
        case DType::kString:
            return "string";
        case DType::kBoolean:
            return "boolean";
    }
    return "unknown";
}

arrow::Result<Column> Column::Make(std::string name, std::shared_ptr<arrow::Array> data) {
    if (!data) {
        return arrow::Status::Invalid("Column data cannot be null.");
    }
    if (!IsSupportedType(data->type())) {
        return arrow::Status::TypeError("Unsupported column type: ", data->type()->ToString());
    }
    ARROW_RETURN_NOT_OK(RejectNaNInFloatingArrays(data));
    const DType dtype=ToDType(data->type());
    return Column(std::move(name), std::move(data), dtype);
}

arrow::Result<Column> ApplyBinaryNumericOp(
    const Column& lhs,
    const Column& rhs,
    BinaryOp op,
    std::string output_name) {
    if (lhs.length()!=rhs.length()) {
        return arrow::Status::Invalid(
            "Binary operations require equal-length columns: left=",
            lhs.length(),
            ", right=",
            rhs.length());
    }
    if (!IsNumericType(lhs.data()->type())||!IsNumericType(rhs.data()->type())) {
        return arrow::Status::TypeError(
            "Incompatible operation: numeric operation requested for non-numeric types ",
            lhs.data()->type()->ToString(),
            " and ",
            rhs.data()->type()->ToString());
    }
    const auto promoted_type=ResolvePromotedNumericType(lhs.data()->type(), rhs.data()->type());
    ARROW_ASSIGN_OR_RAISE(auto result_array, EvaluateNumericBinaryOp(lhs.data(), rhs.data(), promoted_type, op));
    // Enforce null-only missing-value representation even after compute kernels.
    ARROW_RETURN_NOT_OK(RejectNaNInFloatingArrays(result_array));
    return Column::Make(std::move(output_name), std::move(result_array));
}

arrow::Result<Column> ApplyUnaryAbsOp(
    const Column& input,
    std::string output_name) {
    if (!IsNumericType(input.data()->type())) {
        return arrow::Status::TypeError(
            "Incompatible operation: abs requested for non-numeric type ",
            input.data()->type()->ToString());
    }
    if (input.data()->type_id()==arrow::Type::INT32) {
        const auto& arr=static_cast<const arrow::Int32Array&>(*input.data());
        arrow::Int32Builder builder;
        for (int64_t i=0; i<arr.length(); ++i) {
            if (arr.IsNull(i)) {
                ARROW_RETURN_NOT_OK(builder.AppendNull());
            } else {
                ARROW_RETURN_NOT_OK(builder.Append(std::abs(arr.Value(i))));
            }
        }
        std::shared_ptr<arrow::Array> out;
        ARROW_RETURN_NOT_OK(builder.Finish(&out));
        return Column::Make(std::move(output_name), std::move(out));
    }
    if (input.data()->type_id()==arrow::Type::INT64) {
        const auto& arr=static_cast<const arrow::Int64Array&>(*input.data());
        arrow::Int64Builder builder;
        for (int64_t i=0; i<arr.length(); ++i) {
            if (arr.IsNull(i)) {
                ARROW_RETURN_NOT_OK(builder.AppendNull());
            } else {
                ARROW_RETURN_NOT_OK(builder.Append(std::abs(arr.Value(i))));
            }
        }
        std::shared_ptr<arrow::Array> out;
        ARROW_RETURN_NOT_OK(builder.Finish(&out));
        return Column::Make(std::move(output_name), std::move(out));
    }
    if (input.data()->type_id()==arrow::Type::FLOAT) {
        const auto& arr=static_cast<const arrow::FloatArray&>(*input.data());
        arrow::FloatBuilder builder;
        for (int64_t i=0; i<arr.length(); ++i) {
            if (arr.IsNull(i)) {
                ARROW_RETURN_NOT_OK(builder.AppendNull());
            } else {
                ARROW_RETURN_NOT_OK(builder.Append(std::fabs(arr.Value(i))));
            }
        }
        std::shared_ptr<arrow::Array> out;
        ARROW_RETURN_NOT_OK(builder.Finish(&out));
        return Column::Make(std::move(output_name), std::move(out));
    }
    const auto& arr=static_cast<const arrow::DoubleArray&>(*input.data());
    arrow::DoubleBuilder builder;
    for (int64_t i=0; i<arr.length(); ++i) {
        if (arr.IsNull(i)) {
            ARROW_RETURN_NOT_OK(builder.AppendNull());
        } else {
            ARROW_RETURN_NOT_OK(builder.Append(std::fabs(arr.Value(i))));
        }
    }
    std::shared_ptr<arrow::Array> out;
    ARROW_RETURN_NOT_OK(builder.Finish(&out));
    return Column::Make(std::move(output_name), std::move(out));
}

arrow::Result<Column> ApplyComparisonOp(
    const Column& lhs,
    const Column& rhs,
    ComparisonOp op,
    std::string output_name) {
    if (lhs.length()!=rhs.length()) {
        return arrow::Status::Invalid(
            "Comparison operations require equal-length columns: left=",
            lhs.length(),
            ", right=",
            rhs.length());
    }
    const bool both_numeric=IsNumericType(lhs.data()->type())&&IsNumericType(rhs.data()->type());
    const bool same_type=lhs.data()->type_id()==rhs.data()->type_id();
    if (!both_numeric&&!same_type) {
        return arrow::Status::TypeError(
            "Incompatible comparison types: ",
            lhs.data()->type()->ToString(),
            " and ",
            rhs.data()->type()->ToString());
    }
    const bool both_int =
        both_numeric &&
        (lhs.data()->type_id()==arrow::Type::INT32||lhs.data()->type_id()==arrow::Type::INT64) &&
        (rhs.data()->type_id()==arrow::Type::INT32||rhs.data()->type_id()==arrow::Type::INT64);
    arrow::BooleanBuilder builder;
    for (int64_t i=0; i<lhs.length(); ++i) {
        if (lhs.data()->IsNull(i)||rhs.data()->IsNull(i)) {
            ARROW_RETURN_NOT_OK(builder.AppendNull());
            continue;
        }
        bool result=false;
        if (both_int) {
            const int64_t lv=(lhs.data()->type_id()==arrow::Type::INT32)
                ? static_cast<int64_t>(static_cast<const arrow::Int32Array&>(*lhs.data()).Value(i))
                : static_cast<const arrow::Int64Array&>(*lhs.data()).Value(i);
            const int64_t rv=(rhs.data()->type_id()==arrow::Type::INT32)
                ? static_cast<int64_t>(static_cast<const arrow::Int32Array&>(*rhs.data()).Value(i))
                : static_cast<const arrow::Int64Array&>(*rhs.data()).Value(i);
            switch (op) {
                case ComparisonOp::kEqual:        result=(lv==rv); break;
                case ComparisonOp::kNotEqual:     result=(lv!=rv); break;
                case ComparisonOp::kLess:         result=(lv< rv); break;
                case ComparisonOp::kLessEqual:    result=(lv<=rv); break;
                case ComparisonOp::kGreater:      result=(lv> rv); break;
                case ComparisonOp::kGreaterEqual: result=(lv>=rv); break;
            }
        } else if (both_numeric) {
            const double lv=GetNumericAsDouble(lhs.data(), i);
            const double rv=GetNumericAsDouble(rhs.data(), i);
            switch (op) {
                case ComparisonOp::kEqual:        result=(lv==rv); break;
                case ComparisonOp::kNotEqual:     result=(lv!=rv); break;
                case ComparisonOp::kLess:         result=(lv< rv); break;
                case ComparisonOp::kLessEqual:    result=(lv<=rv); break;
                case ComparisonOp::kGreater:      result=(lv> rv); break;
                case ComparisonOp::kGreaterEqual: result=(lv>=rv); break;
            }
        } else if (lhs.data()->type_id()==arrow::Type::STRING) {
            const auto& la=static_cast<const arrow::StringArray&>(*lhs.data());
            const auto& ra=static_cast<const arrow::StringArray&>(*rhs.data());
            const std::string lv=la.GetString(i);
            const std::string rv=ra.GetString(i);
            switch (op) {
                case ComparisonOp::kEqual: result=(lv==rv); break;
                case ComparisonOp::kNotEqual: result=(lv!=rv); break;
                case ComparisonOp::kLess: result=(lv<rv); break;
                case ComparisonOp::kLessEqual: result=(lv<=rv); break;
                case ComparisonOp::kGreater: result=(lv>rv); break;
                case ComparisonOp::kGreaterEqual: result=(lv>=rv); break;
            }
        } else if (lhs.data()->type_id()==arrow::Type::BOOL) {
            const auto& la=static_cast<const arrow::BooleanArray&>(*lhs.data());
            const auto& ra=static_cast<const arrow::BooleanArray&>(*rhs.data());
            const bool lv=la.Value(i);
            const bool rv=ra.Value(i);
            switch (op) {
                case ComparisonOp::kEqual: result=(lv==rv); break;
                case ComparisonOp::kNotEqual: result=(lv!=rv); break;
                case ComparisonOp::kLess: result=(lv<rv); break;
                case ComparisonOp::kLessEqual: result=(lv<=rv); break;
                case ComparisonOp::kGreater: result=(lv>rv); break;
                case ComparisonOp::kGreaterEqual: result=(lv>=rv); break;
            }
        } else {
            return arrow::Status::TypeError(
                "Unsupported comparison type: ",
                lhs.data()->type()->ToString());
        }
        ARROW_RETURN_NOT_OK(builder.Append(result));
    }
    std::shared_ptr<arrow::Array> out;
    ARROW_RETURN_NOT_OK(builder.Finish(&out));
    return Column::Make(std::move(output_name), std::move(out));
}

arrow::Result<Column> ApplyBooleanBinaryOp(
    const Column& lhs,
    const Column& rhs,
    BooleanBinaryOp op,
    std::string output_name) {
    if (lhs.length()!=rhs.length()) {
        return arrow::Status::Invalid(
            "Boolean operations require equal-length columns: left=",
            lhs.length(),
            ", right=",
            rhs.length());
    }
    if (lhs.data()->type_id()!=arrow::Type::BOOL||rhs.data()->type_id()!=arrow::Type::BOOL) {
        return arrow::Status::TypeError(
            "Boolean operation requested for non-boolean types ",
            lhs.data()->type()->ToString(),
            " and ",
            rhs.data()->type()->ToString());
    }
    const auto& la=static_cast<const arrow::BooleanArray&>(*lhs.data());
    const auto& ra=static_cast<const arrow::BooleanArray&>(*rhs.data());
    arrow::BooleanBuilder builder;
    for (int64_t i=0; i<lhs.length(); ++i) {
        if (la.IsNull(i)||ra.IsNull(i)) {
            ARROW_RETURN_NOT_OK(builder.AppendNull());
            continue;
        }
        const bool result=(op==BooleanBinaryOp::kAnd) ? (la.Value(i)&&ra.Value(i)) : (la.Value(i)||ra.Value(i));
        ARROW_RETURN_NOT_OK(builder.Append(result));
    }
    std::shared_ptr<arrow::Array> out;
    ARROW_RETURN_NOT_OK(builder.Finish(&out));
    return Column::Make(std::move(output_name), std::move(out));
}

arrow::Result<Column> ApplyBooleanNotOp(
    const Column& input,
    std::string output_name) {
    if (input.data()->type_id()!=arrow::Type::BOOL) {
        return arrow::Status::TypeError(
            "Boolean NOT requested for non-boolean type ",
            input.data()->type()->ToString());
    }
    const auto& arr=static_cast<const arrow::BooleanArray&>(*input.data());
    arrow::BooleanBuilder builder;
    for (int64_t i=0; i<input.length(); ++i) {
        if (arr.IsNull(i)) {
            ARROW_RETURN_NOT_OK(builder.AppendNull());
        } else {
            ARROW_RETURN_NOT_OK(builder.Append(!arr.Value(i)));
        }
    }
    std::shared_ptr<arrow::Array> out;
    ARROW_RETURN_NOT_OK(builder.Finish(&out));
    return Column::Make(std::move(output_name), std::move(out));
}

arrow::Result<Column> ApplyNullPredicateOp(
    const Column& input,
    NullPredicateOp op,
    std::string output_name) {
    arrow::BooleanBuilder builder;
    for (int64_t i=0; i<input.length(); ++i) {
        const bool is_null=input.data()->IsNull(i);
        const bool result=(op==NullPredicateOp::kIsNull) ? is_null : !is_null;
        ARROW_RETURN_NOT_OK(builder.Append(result));
    }
    std::shared_ptr<arrow::Array> out;
    ARROW_RETURN_NOT_OK(builder.Finish(&out));
    return Column::Make(std::move(output_name), std::move(out));
}

arrow::Result<Column> ApplyStringUnaryOp(
    const Column& input,
    StringUnaryOp op,
    std::string output_name) {
    if (input.data()->type_id()!=arrow::Type::STRING) {
        return arrow::Status::TypeError(
            "String operation requested for non-string type ",
            input.data()->type()->ToString());
    }
    const auto& arr=static_cast<const arrow::StringArray&>(*input.data());
    if (op==StringUnaryOp::kLength) {
        arrow::Int64Builder builder;
        for (int64_t i=0; i<input.length(); ++i) {
            if (arr.IsNull(i)) {
                ARROW_RETURN_NOT_OK(builder.AppendNull());
            } else {
                ARROW_RETURN_NOT_OK(builder.Append(static_cast<int64_t>(arr.GetString(i).size())));
            }
        }
        std::shared_ptr<arrow::Array> out;
        ARROW_RETURN_NOT_OK(builder.Finish(&out));
        return Column::Make(std::move(output_name), std::move(out));
    }
    auto normalize_case=[op](std::string value) {
        for (char& ch : value) {
            const unsigned char uch=static_cast<unsigned char>(ch);
            ch=static_cast<char>(op==StringUnaryOp::kToLower ? std::tolower(uch) : std::toupper(uch));
        }
        return value;
    };
    arrow::StringBuilder builder;
    for (int64_t i=0; i<input.length(); ++i) {
        if (arr.IsNull(i)) {
            ARROW_RETURN_NOT_OK(builder.AppendNull());
        } else {
            ARROW_RETURN_NOT_OK(builder.Append(normalize_case(arr.GetString(i))));
        }
    }
    std::shared_ptr<arrow::Array> out;
    ARROW_RETURN_NOT_OK(builder.Finish(&out));
    return Column::Make(std::move(output_name), std::move(out));
}

arrow::Result<Column> ApplyStringPredicateOp(
    const Column& input,
    StringPredicateOp op,
    std::string pattern,
    std::string output_name) {
    if (input.data()->type_id()!=arrow::Type::STRING) {
        return arrow::Status::TypeError(
            "String predicate requested for non-string type ",
            input.data()->type()->ToString());
    }
    const auto& arr=static_cast<const arrow::StringArray&>(*input.data());
    arrow::BooleanBuilder builder;
    for (int64_t i=0; i<input.length(); ++i) {
        if (arr.IsNull(i)) {
            ARROW_RETURN_NOT_OK(builder.AppendNull());
            continue;
        }
        const std::string value=arr.GetString(i);
        bool result=false;
        switch (op) {
            case StringPredicateOp::kContains:
                result=value.find(pattern)!=std::string::npos;
                break;
            case StringPredicateOp::kStartsWith:
                result=value.rfind(pattern, 0)==0;
                break;
            case StringPredicateOp::kEndsWith:
                result=value.size()>=pattern.size() &&
                         value.compare(value.size()-pattern.size(), pattern.size(), pattern)==0;
                break;
        }
        ARROW_RETURN_NOT_OK(builder.Append(result));
    }
    std::shared_ptr<arrow::Array> out;
    ARROW_RETURN_NOT_OK(builder.Finish(&out));
    return Column::Make(std::move(output_name), std::move(out));
}

}  
