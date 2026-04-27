#include "dataframelib/eager_dataframe.hpp"
#include <algorithm>
#include <cstring>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <arrow/array/concatenate.h>
#include <arrow/csv/api.h>
#include <arrow/csv/writer.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

namespace dataframelib {

namespace {

arrow::Result<std::vector<Column>> BuildValidatedColumns(const EagerDataFrame::ColumnMap& input_columns) {
    std::vector<std::string> names;
    names.reserve(input_columns.size());
    for (const auto& kv : input_columns) {
        names.push_back(kv.first);
    }
    std::sort(names.begin(), names.end());

    std::vector<Column> validated;
    validated.reserve(names.size());

    int64_t expected_length = -1;
    for (const auto& name : names) {
        auto iter = input_columns.find(name);
        if (iter == input_columns.end()) {
            return arrow::Status::Invalid("Column map lookup failed for name: ", name);
        }

        ARROW_ASSIGN_OR_RAISE(auto col, Column::Make(name, iter->second));

        if (expected_length < 0) {
            expected_length = col.length();
        } else if (col.length() != expected_length) {
            return arrow::Status::Invalid(
                "All columns must have equal length. Column '",
                name,
                "' length=",
                col.length(),
                " expected=",
                expected_length);
        }

        validated.push_back(std::move(col));
    }

    return validated;
}

std::shared_ptr<arrow::Schema> BuildSchema(const std::vector<Column>& columns) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    fields.reserve(columns.size());

    for (const auto& col : columns) {
        fields.push_back(arrow::field(col.name(), col.data()->type(), true));
    }

    return arrow::schema(std::move(fields));
}

arrow::Result<std::shared_ptr<arrow::Array>> ToSingleArray(const std::shared_ptr<arrow::ChunkedArray>& chunked) {
    if (!chunked) {
        return arrow::Status::Invalid("ChunkedArray cannot be null.");
    }

    if (chunked->num_chunks() == 1) {
        return chunked->chunk(0);
    }

    ARROW_ASSIGN_OR_RAISE(auto merged, arrow::Concatenate(chunked->chunks(), arrow::default_memory_pool()));
    return merged;
}

std::string DefaultExprName(int64_t index) {
    std::ostringstream oss;
    oss << "expr_" << index;
    return oss.str();
}

arrow::Result<std::shared_ptr<arrow::Array>> BroadcastScalarToArray(
    const std::shared_ptr<arrow::Scalar>& scalar,
    int64_t length) {
    if (!scalar) {
        return arrow::Status::Invalid("Literal scalar cannot be null.");
    }

    switch (scalar->type->id()) {
        case arrow::Type::INT32: {
            arrow::Int32Builder builder;
            const auto& s = static_cast<const arrow::Int32Scalar&>(*scalar);
            for (int64_t i = 0; i < length; ++i) {
                ARROW_RETURN_NOT_OK(s.is_valid ? builder.Append(s.value) : builder.AppendNull());
            }
            std::shared_ptr<arrow::Array> out;
            ARROW_RETURN_NOT_OK(builder.Finish(&out));
            return out;
        }
        case arrow::Type::INT64: {
            arrow::Int64Builder builder;
            const auto& s = static_cast<const arrow::Int64Scalar&>(*scalar);
            for (int64_t i = 0; i < length; ++i) {
                ARROW_RETURN_NOT_OK(s.is_valid ? builder.Append(s.value) : builder.AppendNull());
            }
            std::shared_ptr<arrow::Array> out;
            ARROW_RETURN_NOT_OK(builder.Finish(&out));
            return out;
        }
        case arrow::Type::FLOAT: {
            arrow::FloatBuilder builder;
            const auto& s = static_cast<const arrow::FloatScalar&>(*scalar);
            for (int64_t i = 0; i < length; ++i) {
                ARROW_RETURN_NOT_OK(s.is_valid ? builder.Append(s.value) : builder.AppendNull());
            }
            std::shared_ptr<arrow::Array> out;
            ARROW_RETURN_NOT_OK(builder.Finish(&out));
            return out;
        }
        case arrow::Type::DOUBLE: {
            arrow::DoubleBuilder builder;
            const auto& s = static_cast<const arrow::DoubleScalar&>(*scalar);
            for (int64_t i = 0; i < length; ++i) {
                ARROW_RETURN_NOT_OK(s.is_valid ? builder.Append(s.value) : builder.AppendNull());
            }
            std::shared_ptr<arrow::Array> out;
            ARROW_RETURN_NOT_OK(builder.Finish(&out));
            return out;
        }
        case arrow::Type::BOOL: {
            arrow::BooleanBuilder builder;
            const auto& s = static_cast<const arrow::BooleanScalar&>(*scalar);
            for (int64_t i = 0; i < length; ++i) {
                ARROW_RETURN_NOT_OK(s.is_valid ? builder.Append(s.value) : builder.AppendNull());
            }
            std::shared_ptr<arrow::Array> out;
            ARROW_RETURN_NOT_OK(builder.Finish(&out));
            return out;
        }
        case arrow::Type::STRING: {
            arrow::StringBuilder builder;
            const auto& s = static_cast<const arrow::StringScalar&>(*scalar);
            const std::string str_val = s.is_valid
                ? std::string(reinterpret_cast<const char*>(s.value->data()), s.value->size())
                : std::string{};
            for (int64_t i = 0; i < length; ++i) {
                ARROW_RETURN_NOT_OK(s.is_valid ? builder.Append(str_val) : builder.AppendNull());
            }
            std::shared_ptr<arrow::Array> out;
            ARROW_RETURN_NOT_OK(builder.Finish(&out));
            return out;
        }
        default:
            return arrow::Status::TypeError("Unsupported literal type: ", scalar->type->ToString());
    }
}

arrow::Result<std::shared_ptr<arrow::Array>> FilterArrayByMask(
    const std::shared_ptr<arrow::Array>& data,
    const arrow::BooleanArray& mask) {
    if (data->length() != mask.length()) {
        return arrow::Status::Invalid("Filter mask length mismatch.");
    }

    switch (data->type_id()) {
        case arrow::Type::INT32: {
            const auto& arr = static_cast<const arrow::Int32Array&>(*data);
            arrow::Int32Builder builder;
            for (int64_t i = 0; i < arr.length(); ++i) {
                if (!mask.IsNull(i) && mask.Value(i)) {
                    ARROW_RETURN_NOT_OK(arr.IsNull(i) ? builder.AppendNull() : builder.Append(arr.Value(i)));
                }
            }
            std::shared_ptr<arrow::Array> out;
            ARROW_RETURN_NOT_OK(builder.Finish(&out));
            return out;
        }
        case arrow::Type::INT64: {
            const auto& arr = static_cast<const arrow::Int64Array&>(*data);
            arrow::Int64Builder builder;
            for (int64_t i = 0; i < arr.length(); ++i) {
                if (!mask.IsNull(i) && mask.Value(i)) {
                    ARROW_RETURN_NOT_OK(arr.IsNull(i) ? builder.AppendNull() : builder.Append(arr.Value(i)));
                }
            }
            std::shared_ptr<arrow::Array> out;
            ARROW_RETURN_NOT_OK(builder.Finish(&out));
            return out;
        }
        case arrow::Type::FLOAT: {
            const auto& arr = static_cast<const arrow::FloatArray&>(*data);
            arrow::FloatBuilder builder;
            for (int64_t i = 0; i < arr.length(); ++i) {
                if (!mask.IsNull(i) && mask.Value(i)) {
                    ARROW_RETURN_NOT_OK(arr.IsNull(i) ? builder.AppendNull() : builder.Append(arr.Value(i)));
                }
            }
            std::shared_ptr<arrow::Array> out;
            ARROW_RETURN_NOT_OK(builder.Finish(&out));
            return out;
        }
        case arrow::Type::DOUBLE: {
            const auto& arr = static_cast<const arrow::DoubleArray&>(*data);
            arrow::DoubleBuilder builder;
            for (int64_t i = 0; i < arr.length(); ++i) {
                if (!mask.IsNull(i) && mask.Value(i)) {
                    ARROW_RETURN_NOT_OK(arr.IsNull(i) ? builder.AppendNull() : builder.Append(arr.Value(i)));
                }
            }
            std::shared_ptr<arrow::Array> out;
            ARROW_RETURN_NOT_OK(builder.Finish(&out));
            return out;
        }
        case arrow::Type::STRING: {
            const auto& arr = static_cast<const arrow::StringArray&>(*data);
            arrow::StringBuilder builder;
            for (int64_t i = 0; i < arr.length(); ++i) {
                if (!mask.IsNull(i) && mask.Value(i)) {
                    ARROW_RETURN_NOT_OK(arr.IsNull(i) ? builder.AppendNull() : builder.Append(arr.GetString(i)));
                }
            }
            std::shared_ptr<arrow::Array> out;
            ARROW_RETURN_NOT_OK(builder.Finish(&out));
            return out;
        }
        case arrow::Type::BOOL: {
            const auto& arr = static_cast<const arrow::BooleanArray&>(*data);
            arrow::BooleanBuilder builder;
            for (int64_t i = 0; i < arr.length(); ++i) {
                if (!mask.IsNull(i) && mask.Value(i)) {
                    ARROW_RETURN_NOT_OK(arr.IsNull(i) ? builder.AppendNull() : builder.Append(arr.Value(i)));
                }
            }
            std::shared_ptr<arrow::Array> out;
            ARROW_RETURN_NOT_OK(builder.Finish(&out));
            return out;
        }
        default:
            return arrow::Status::TypeError("Unsupported column type for filter: ", data->type()->ToString());
    }
}

arrow::Result<Column> EvaluateNodeToColumn(
    const EagerDataFrame& df,
    const std::shared_ptr<ExprNode>& node_ptr,
    const std::string& fallback_name) {
    if (!node_ptr) {
        return arrow::Status::Invalid("Expression node cannot be null.");
    }

    const ExprNode& node = *node_ptr;
    switch (node.kind()) {
        case ExprNode::Kind::Col: {
            const auto& col_node = static_cast<const ColNode&>(node);
            ARROW_ASSIGN_OR_RAISE(const Column* source, df.column(col_node.name()));
            return Column::Make(fallback_name, source->data());
        }
        case ExprNode::Kind::Lit: {
            const auto& lit_node = static_cast<const LitNode&>(node);
            ARROW_ASSIGN_OR_RAISE(auto array, BroadcastScalarToArray(lit_node.scalar(), df.num_rows()));
            return Column::Make(fallback_name, std::move(array));
        }
        case ExprNode::Kind::Alias: {
            const auto& alias_node = static_cast<const AliasNode&>(node);
            return EvaluateNodeToColumn(df, alias_node.input(), alias_node.name());
        }
        case ExprNode::Kind::Binary: {
            const auto& b = static_cast<const BinaryNode&>(node);
            ARROW_ASSIGN_OR_RAISE(Column left, EvaluateNodeToColumn(df, b.left(), fallback_name + "_l"));
            ARROW_ASSIGN_OR_RAISE(Column right, EvaluateNodeToColumn(df, b.right(), fallback_name + "_r"));

            switch (b.op()) {
                case BinaryOpKind::kAdd:
                    return ApplyBinaryNumericOp(left, right, BinaryOp::kAdd, fallback_name);
                case BinaryOpKind::kSub:
                    return ApplyBinaryNumericOp(left, right, BinaryOp::kSubtract, fallback_name);
                case BinaryOpKind::kMul:
                    return ApplyBinaryNumericOp(left, right, BinaryOp::kMultiply, fallback_name);
                case BinaryOpKind::kDiv:
                    return ApplyBinaryNumericOp(left, right, BinaryOp::kDivide, fallback_name);
                case BinaryOpKind::kMod:
                    return ApplyBinaryNumericOp(left, right, BinaryOp::kModulo, fallback_name);
                case BinaryOpKind::kEq:
                    return ApplyComparisonOp(left, right, ComparisonOp::kEqual, fallback_name);
                case BinaryOpKind::kNeq:
                    return ApplyComparisonOp(left, right, ComparisonOp::kNotEqual, fallback_name);
                case BinaryOpKind::kLt:
                    return ApplyComparisonOp(left, right, ComparisonOp::kLess, fallback_name);
                case BinaryOpKind::kLe:
                    return ApplyComparisonOp(left, right, ComparisonOp::kLessEqual, fallback_name);
                case BinaryOpKind::kGt:
                    return ApplyComparisonOp(left, right, ComparisonOp::kGreater, fallback_name);
                case BinaryOpKind::kGe:
                    return ApplyComparisonOp(left, right, ComparisonOp::kGreaterEqual, fallback_name);
                case BinaryOpKind::kAnd:
                    return ApplyBooleanBinaryOp(left, right, BooleanBinaryOp::kAnd, fallback_name);
                case BinaryOpKind::kOr:
                    return ApplyBooleanBinaryOp(left, right, BooleanBinaryOp::kOr, fallback_name);
            }
            return arrow::Status::NotImplemented("Unsupported binary expression op.");
        }
        case ExprNode::Kind::Unary: {
            const auto& u = static_cast<const UnaryNode&>(node);
            ARROW_ASSIGN_OR_RAISE(Column input, EvaluateNodeToColumn(df, u.input(), fallback_name + "_u"));
            switch (u.op()) {
                case UnaryOpKind::kNot:
                    return ApplyBooleanNotOp(input, fallback_name);
                case UnaryOpKind::kAbs:
                    return ApplyUnaryAbsOp(input, fallback_name);
                case UnaryOpKind::kIsNull:
                    return ApplyNullPredicateOp(input, NullPredicateOp::kIsNull, fallback_name);
                case UnaryOpKind::kIsNotNull:
                    return ApplyNullPredicateOp(input, NullPredicateOp::kIsNotNull, fallback_name);
            }
            return arrow::Status::NotImplemented("Unsupported unary expression op.");
        }
        case ExprNode::Kind::StringOp: {
            const auto& s = static_cast<const StringOpNode&>(node);
            ARROW_ASSIGN_OR_RAISE(Column input, EvaluateNodeToColumn(df, s.input(), fallback_name + "_s"));
            switch (s.op()) {
                case StringOpKind::kLength:
                    return ApplyStringUnaryOp(input, StringUnaryOp::kLength, fallback_name);
                case StringOpKind::kContains:
                    return ApplyStringPredicateOp(input, StringPredicateOp::kContains, s.arg(), fallback_name);
                case StringOpKind::kStartsWith:
                    return ApplyStringPredicateOp(input, StringPredicateOp::kStartsWith, s.arg(), fallback_name);
                case StringOpKind::kEndsWith:
                    return ApplyStringPredicateOp(input, StringPredicateOp::kEndsWith, s.arg(), fallback_name);
                case StringOpKind::kToLower:
                    return ApplyStringUnaryOp(input, StringUnaryOp::kToLower, fallback_name);
                case StringOpKind::kToUpper:
                    return ApplyStringUnaryOp(input, StringUnaryOp::kToUpper, fallback_name);
            }
            return arrow::Status::NotImplemented("Unsupported string expression op.");
        }
        case ExprNode::Kind::Agg:
            return arrow::Status::Invalid(
                "Aggregate expressions are only valid inside aggregate(); "
                "use group_by().aggregate() instead.");
    }
    return arrow::Status::NotImplemented("Unsupported expression kind.");
}

arrow::Result<Column> EvaluateExprToColumn(
    const EagerDataFrame& df,
    const Expr& expr,
    const std::string& fallback_name) {
    return EvaluateNodeToColumn(df, expr.node_ptr(), fallback_name);
}

std::string OutputNameForExpr(const Expr& expr, int64_t index) {
    if (expr.node().kind() == ExprNode::Kind::Alias) {
        const auto& alias = static_cast<const AliasNode&>(expr.node());
        return alias.name();
    }
    if (expr.node().kind() == ExprNode::Kind::Col) {
        const auto& col_node = static_cast<const ColNode&>(expr.node());
        return col_node.name();
    }
    return DefaultExprName(index);
}

std::string CellToKeyString(const std::shared_ptr<arrow::Array>& arr, int64_t row) {
    const std::string type_tag = "t" + std::to_string(static_cast<int>(arr->type_id())) + ":";
    if (arr->IsNull(row)) return type_tag + "\x01";
    switch (arr->type_id()) {
        case arrow::Type::INT32:
            return type_tag + "\x02" + std::to_string(static_cast<const arrow::Int32Array&>(*arr).Value(row));
        case arrow::Type::INT64:
            return type_tag + "\x02" + std::to_string(static_cast<const arrow::Int64Array&>(*arr).Value(row));
        case arrow::Type::FLOAT: {
            float v = static_cast<const arrow::FloatArray&>(*arr).Value(row);
            if (v == 0.0f) v = 0.0f;
            uint32_t bits = 0;
            std::memcpy(&bits, &v, sizeof(bits));
            return type_tag + "\x02" + std::to_string(bits);
        }
        case arrow::Type::DOUBLE: {
            double v = static_cast<const arrow::DoubleArray&>(*arr).Value(row);
            if (v == 0.0) v = 0.0;
            uint64_t bits = 0;
            std::memcpy(&bits, &v, sizeof(bits));
            return type_tag + "\x02" + std::to_string(bits);
        }
        case arrow::Type::BOOL:
            return type_tag + "\x02" + std::string(static_cast<const arrow::BooleanArray&>(*arr).Value(row) ? "1" : "0");
        case arrow::Type::STRING:
            return type_tag + "\x02" + static_cast<const arrow::StringArray&>(*arr).GetString(row);
        default:
            return type_tag + "\x02?";
    }
}

arrow::Result<std::shared_ptr<arrow::Array>> GatherArray(
    const std::shared_ptr<arrow::Array>& data,
    const std::vector<int64_t>& indices) {
    switch (data->type_id()) {
        case arrow::Type::INT32: {
            const auto& src = static_cast<const arrow::Int32Array&>(*data);
            arrow::Int32Builder b;
            for (int64_t i : indices)
                ARROW_RETURN_NOT_OK(i < 0 || src.IsNull(i) ? b.AppendNull() : b.Append(src.Value(i)));
            std::shared_ptr<arrow::Array> out; ARROW_RETURN_NOT_OK(b.Finish(&out)); return out;
        }
        case arrow::Type::INT64: {
            const auto& src = static_cast<const arrow::Int64Array&>(*data);
            arrow::Int64Builder b;
            for (int64_t i : indices)
                ARROW_RETURN_NOT_OK(i < 0 || src.IsNull(i) ? b.AppendNull() : b.Append(src.Value(i)));
            std::shared_ptr<arrow::Array> out; ARROW_RETURN_NOT_OK(b.Finish(&out)); return out;
        }
        case arrow::Type::FLOAT: {
            const auto& src = static_cast<const arrow::FloatArray&>(*data);
            arrow::FloatBuilder b;
            for (int64_t i : indices)
                ARROW_RETURN_NOT_OK(i < 0 || src.IsNull(i) ? b.AppendNull() : b.Append(src.Value(i)));
            std::shared_ptr<arrow::Array> out; ARROW_RETURN_NOT_OK(b.Finish(&out)); return out;
        }
        case arrow::Type::DOUBLE: {
            const auto& src = static_cast<const arrow::DoubleArray&>(*data);
            arrow::DoubleBuilder b;
            for (int64_t i : indices)
                ARROW_RETURN_NOT_OK(i < 0 || src.IsNull(i) ? b.AppendNull() : b.Append(src.Value(i)));
            std::shared_ptr<arrow::Array> out; ARROW_RETURN_NOT_OK(b.Finish(&out)); return out;
        }
        case arrow::Type::STRING: {
            const auto& src = static_cast<const arrow::StringArray&>(*data);
            arrow::StringBuilder b;
            for (int64_t i : indices)
                ARROW_RETURN_NOT_OK(i < 0 || src.IsNull(i) ? b.AppendNull() : b.Append(src.GetString(i)));
            std::shared_ptr<arrow::Array> out; ARROW_RETURN_NOT_OK(b.Finish(&out)); return out;
        }
        case arrow::Type::BOOL: {
            const auto& src = static_cast<const arrow::BooleanArray&>(*data);
            arrow::BooleanBuilder b;
            for (int64_t i : indices)
                ARROW_RETURN_NOT_OK(i < 0 || src.IsNull(i) ? b.AppendNull() : b.Append(src.Value(i)));
            std::shared_ptr<arrow::Array> out; ARROW_RETURN_NOT_OK(b.Finish(&out)); return out;
        }
        default:
            return arrow::Status::TypeError("GatherArray: unsupported type: ", data->type()->ToString());
    }
}

arrow::Result<std::shared_ptr<arrow::Array>> AggregateArrayByGroups(
    const std::shared_ptr<arrow::Array>& data,
    const std::vector<std::vector<int64_t>>& groups,
    AggOpKind op) {

    if (op == AggOpKind::kCount) {
        arrow::Int64Builder b;
        for (const auto& grp : groups) {
            int64_t cnt = 0;
            for (int64_t i : grp) if (!data->IsNull(i)) ++cnt;
            ARROW_RETURN_NOT_OK(b.Append(cnt));
        }
        std::shared_ptr<arrow::Array> out; ARROW_RETURN_NOT_OK(b.Finish(&out)); return out;
    }

    if (op == AggOpKind::kMean) {
        arrow::DoubleBuilder b;
        for (const auto& grp : groups) {
            double sum = 0.0; int64_t cnt = 0;
            for (int64_t i : grp) {
                if (data->IsNull(i)) continue;
                switch (data->type_id()) {
                    case arrow::Type::INT32:  sum += static_cast<const arrow::Int32Array&>(*data).Value(i);  break;
                    case arrow::Type::INT64:  sum += static_cast<const arrow::Int64Array&>(*data).Value(i);  break;
                    case arrow::Type::FLOAT:  sum += static_cast<const arrow::FloatArray&>(*data).Value(i);  break;
                    case arrow::Type::DOUBLE: sum += static_cast<const arrow::DoubleArray&>(*data).Value(i); break;
                    default:
                        return arrow::Status::TypeError("mean() not supported for type: ", data->type()->ToString());
                }
                ++cnt;
            }
            ARROW_RETURN_NOT_OK(cnt > 0 ? b.Append(sum / cnt) : b.AppendNull());
        }
        std::shared_ptr<arrow::Array> out; ARROW_RETURN_NOT_OK(b.Finish(&out)); return out;
    }

    if (op == AggOpKind::kSum) {
        switch (data->type_id()) {
            case arrow::Type::INT32:
            case arrow::Type::INT64: {
                arrow::Int64Builder b;
                for (const auto& grp : groups) {
                    int64_t sum = 0; bool has = false;
                    for (int64_t i : grp) {
                        if (data->IsNull(i)) continue;
                        sum += (data->type_id() == arrow::Type::INT32)
                            ? static_cast<const arrow::Int32Array&>(*data).Value(i)
                            : static_cast<const arrow::Int64Array&>(*data).Value(i);
                        has = true;
                    }
                    ARROW_RETURN_NOT_OK(has ? b.Append(sum) : b.AppendNull());
                }
                std::shared_ptr<arrow::Array> out; ARROW_RETURN_NOT_OK(b.Finish(&out)); return out;
            }
            case arrow::Type::FLOAT:
            case arrow::Type::DOUBLE: {
                arrow::DoubleBuilder b;
                for (const auto& grp : groups) {
                    double sum = 0.0; bool has = false;
                    for (int64_t i : grp) {
                        if (data->IsNull(i)) continue;
                        sum += (data->type_id() == arrow::Type::FLOAT)
                            ? static_cast<const arrow::FloatArray&>(*data).Value(i)
                            : static_cast<const arrow::DoubleArray&>(*data).Value(i);
                        has = true;
                    }
                    ARROW_RETURN_NOT_OK(has ? b.Append(sum) : b.AppendNull());
                }
                std::shared_ptr<arrow::Array> out; ARROW_RETURN_NOT_OK(b.Finish(&out)); return out;
            }
            default:
                return arrow::Status::TypeError("sum() not supported for type: ", data->type()->ToString());
        }
    }

    if (op == AggOpKind::kMin || op == AggOpKind::kMax) {
        const bool is_min = (op == AggOpKind::kMin);
        switch (data->type_id()) {
            case arrow::Type::INT32: {
                const auto& src = static_cast<const arrow::Int32Array&>(*data);
                arrow::Int32Builder b;
                for (const auto& grp : groups) {
                    std::optional<int32_t> val;
                    for (int64_t i : grp) {
                        if (src.IsNull(i)) continue;
                        if (!val || (is_min ? src.Value(i) < *val : src.Value(i) > *val)) val = src.Value(i);
                    }
                    ARROW_RETURN_NOT_OK(val ? b.Append(*val) : b.AppendNull());
                }
                std::shared_ptr<arrow::Array> out; ARROW_RETURN_NOT_OK(b.Finish(&out)); return out;
            }
            case arrow::Type::INT64: {
                const auto& src = static_cast<const arrow::Int64Array&>(*data);
                arrow::Int64Builder b;
                for (const auto& grp : groups) {
                    std::optional<int64_t> val;
                    for (int64_t i : grp) {
                        if (src.IsNull(i)) continue;
                        if (!val || (is_min ? src.Value(i) < *val : src.Value(i) > *val)) val = src.Value(i);
                    }
                    ARROW_RETURN_NOT_OK(val ? b.Append(*val) : b.AppendNull());
                }
                std::shared_ptr<arrow::Array> out; ARROW_RETURN_NOT_OK(b.Finish(&out)); return out;
            }
            case arrow::Type::FLOAT: {
                const auto& src = static_cast<const arrow::FloatArray&>(*data);
                arrow::FloatBuilder b;
                for (const auto& grp : groups) {
                    std::optional<float> val;
                    for (int64_t i : grp) {
                        if (src.IsNull(i)) continue;
                        if (!val || (is_min ? src.Value(i) < *val : src.Value(i) > *val)) val = src.Value(i);
                    }
                    ARROW_RETURN_NOT_OK(val ? b.Append(*val) : b.AppendNull());
                }
                std::shared_ptr<arrow::Array> out; ARROW_RETURN_NOT_OK(b.Finish(&out)); return out;
            }
            case arrow::Type::DOUBLE: {
                const auto& src = static_cast<const arrow::DoubleArray&>(*data);
                arrow::DoubleBuilder b;
                for (const auto& grp : groups) {
                    std::optional<double> val;
                    for (int64_t i : grp) {
                        if (src.IsNull(i)) continue;
                        if (!val || (is_min ? src.Value(i) < *val : src.Value(i) > *val)) val = src.Value(i);
                    }
                    ARROW_RETURN_NOT_OK(val ? b.Append(*val) : b.AppendNull());
                }
                std::shared_ptr<arrow::Array> out; ARROW_RETURN_NOT_OK(b.Finish(&out)); return out;
            }
            case arrow::Type::BOOL: {
                const auto& src = static_cast<const arrow::BooleanArray&>(*data);
                arrow::BooleanBuilder b;
                for (const auto& grp : groups) {
                    std::optional<bool> val;
                    for (int64_t i : grp) {
                        if (src.IsNull(i)) continue;
                        bool v = src.Value(i);
                        if (!val || (is_min ? (v < *val) : (v > *val))) val = v;
                    }
                    ARROW_RETURN_NOT_OK(val ? b.Append(*val) : b.AppendNull());
                }
                std::shared_ptr<arrow::Array> out; ARROW_RETURN_NOT_OK(b.Finish(&out)); return out;
            }
            case arrow::Type::STRING: {
                const auto& src = static_cast<const arrow::StringArray&>(*data);
                arrow::StringBuilder b;
                for (const auto& grp : groups) {
                    std::optional<std::string> val;
                    for (int64_t i : grp) {
                        if (src.IsNull(i)) continue;
                        std::string s = src.GetString(i);
                        if (!val || (is_min ? s < *val : s > *val)) val = s;
                    }
                    ARROW_RETURN_NOT_OK(val ? b.Append(*val) : b.AppendNull());
                }
                std::shared_ptr<arrow::Array> out; ARROW_RETURN_NOT_OK(b.Finish(&out)); return out;
            }
            default:
                return arrow::Status::TypeError("min/max not supported for type: ", data->type()->ToString());
        }
    }

    return arrow::Status::NotImplemented("Unsupported aggregation operation.");
}

} 

arrow::Result<EagerDataFrame> EagerDataFrame::from_arrow_table(const std::shared_ptr<arrow::Table>& table) {
    if (!table) {
        return arrow::Status::Invalid("Arrow table cannot be null.");
    }

    std::vector<Column> columns;
    columns.reserve(table->num_columns());
    for (int i = 0; i < table->num_columns(); ++i) {
        const std::string& name = table->field(i)->name();
        ARROW_ASSIGN_OR_RAISE(auto array, ToSingleArray(table->column(i)));
        ARROW_ASSIGN_OR_RAISE(auto col, Column::Make(name, std::move(array)));
        columns.push_back(std::move(col));
    }

    return EagerDataFrame::from_ordered_columns(std::move(columns));
}

EagerDataFrame EagerDataFrame::from_columns(const ColumnMap& columns) {
    auto helper = [&]() -> arrow::Result<EagerDataFrame> {
        ARROW_ASSIGN_OR_RAISE(auto validated_columns, BuildValidatedColumns(columns));
        const int64_t num_rows = validated_columns.empty() ? 0 : validated_columns.front().length();
        auto schema = BuildSchema(validated_columns);
        return EagerDataFrame(std::move(validated_columns), std::move(schema), num_rows);
    };
    return helper().ValueOrDie();
}

EagerDataFrame EagerDataFrame::from_ordered_columns(std::vector<Column> cols) {
    auto helper = [&]() -> arrow::Result<EagerDataFrame> {
        const int64_t n = cols.empty() ? 0 : cols.front().length();
        std::unordered_set<std::string> seen;
        for (const auto& c : cols) {
            if (c.length() != n) {
                return arrow::Status::Invalid(
                    "All columns must have equal length. Column '", c.name(),
                    "' length=", c.length(), " expected=", n);
            }
            if (!seen.insert(c.name()).second) {
                return arrow::Status::Invalid("Duplicate column name: '", c.name(), "'");
            }
        }
        auto schema = BuildSchema(cols);
        return EagerDataFrame(std::move(cols), std::move(schema), n);
    };
    return helper().ValueOrDie();
}

EagerDataFrame EagerDataFrame::read_csv(const std::string& path) {
    auto helper = [&]() -> arrow::Result<EagerDataFrame> {
        ARROW_ASSIGN_OR_RAISE(auto input, arrow::io::ReadableFile::Open(path));

        auto read_options = arrow::csv::ReadOptions::Defaults();
        auto parse_options = arrow::csv::ParseOptions::Defaults();
        auto convert_options = arrow::csv::ConvertOptions::Defaults();

        arrow::io::IOContext io_context(arrow::default_memory_pool());
        ARROW_ASSIGN_OR_RAISE(
            auto reader,
            arrow::csv::TableReader::Make(
                io_context,
                input,
                read_options,
                parse_options,
                convert_options));
        ARROW_ASSIGN_OR_RAISE(auto table, reader->Read());

        return from_arrow_table(table);
    };
    return helper().ValueOrDie();
}

EagerDataFrame EagerDataFrame::read_parquet(const std::string& path) {
    auto helper = [&]() -> arrow::Result<EagerDataFrame> {
        ARROW_ASSIGN_OR_RAISE(auto input, arrow::io::ReadableFile::Open(path));

        std::unique_ptr<parquet::arrow::FileReader> reader;
        PARQUET_ASSIGN_OR_THROW(
            reader,
            parquet::arrow::OpenFile(input, arrow::default_memory_pool()));

        std::shared_ptr<arrow::Table> table;
        PARQUET_THROW_NOT_OK(reader->ReadTable(&table));

        return from_arrow_table(table);
    };
    return helper().ValueOrDie();
}

arrow::Status EagerDataFrame::write_csv(const std::string& path) const {
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    arrays.reserve(columns_.size());
    for (const auto& col : columns_) {
        arrays.push_back(col.data());
    }

    auto table = arrow::Table::Make(schema_, arrays, num_rows_);

    ARROW_ASSIGN_OR_RAISE(auto output, arrow::io::FileOutputStream::Open(path));
    auto write_options = arrow::csv::WriteOptions::Defaults();
    ARROW_RETURN_NOT_OK(arrow::csv::WriteCSV(*table, write_options, output.get()));
    return output->Close();
}

arrow::Status EagerDataFrame::write_parquet(const std::string& path) const {
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    arrays.reserve(columns_.size());
    for (const auto& col : columns_) {
        arrays.push_back(col.data());
    }

    auto table = arrow::Table::Make(schema_, arrays, num_rows_);

    ARROW_ASSIGN_OR_RAISE(auto output, arrow::io::FileOutputStream::Open(path));
    ARROW_RETURN_NOT_OK(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), output, 1024));
    return output->Close();
}

arrow::Result<const Column*> EagerDataFrame::column(const std::string& name) const {
    for (const auto& col : columns_) {
        if (col.name() == name) {
            return &col;
        }
    }

    return arrow::Status::KeyError("Column not found: ", name);
}

EagerDataFrame EagerDataFrame::select(const std::vector<std::string>& columns) const {
    auto helper = [&]() -> arrow::Result<EagerDataFrame> {
        std::vector<Column> selected;
        selected.reserve(columns.size());
        std::unordered_set<std::string> seen;
        for (const auto& name : columns) {
            if (!seen.insert(name).second) {
                return arrow::Status::Invalid("Duplicate column in select: ", name);
            }
            ARROW_ASSIGN_OR_RAISE(const Column* col_ptr, column(name));
            ARROW_ASSIGN_OR_RAISE(auto col, Column::Make(name, col_ptr->data()));
            selected.push_back(std::move(col));
        }
        return from_ordered_columns(std::move(selected));
    };
    return helper().ValueOrDie();
}

EagerDataFrame EagerDataFrame::select(const std::vector<Expr>& expressions) const {
    auto helper = [&]() -> arrow::Result<EagerDataFrame> {
        std::vector<Column> selected;
        selected.reserve(expressions.size());
        std::unordered_set<std::string> seen;
        for (int64_t i = 0; i < static_cast<int64_t>(expressions.size()); ++i) {
            const auto& expr = expressions[static_cast<size_t>(i)];
            const auto output_name = OutputNameForExpr(expr, i);
            if (!seen.insert(output_name).second) {
                return arrow::Status::Invalid("Duplicate output name in select expressions: ", output_name);
            }
            ARROW_ASSIGN_OR_RAISE(auto col, EvaluateExprToColumn(*this, expr, output_name));
            selected.push_back(std::move(col));
        }
        return from_ordered_columns(std::move(selected));
    };
    return helper().ValueOrDie();
}

EagerDataFrame EagerDataFrame::filter(const Expr& predicate) const {
    auto helper = [&]() -> arrow::Result<EagerDataFrame> {
        ARROW_ASSIGN_OR_RAISE(auto pred_col, EvaluateExprToColumn(*this, predicate, "_predicate"));
        if (pred_col.data()->type_id() != arrow::Type::BOOL) {
            return arrow::Status::TypeError("Filter predicate must evaluate to boolean type.");
        }

        const auto& mask = static_cast<const arrow::BooleanArray&>(*pred_col.data());
        std::vector<Column> filtered;
        filtered.reserve(columns_.size());
        for (const auto& col : columns_) {
            ARROW_ASSIGN_OR_RAISE(auto arr, FilterArrayByMask(col.data(), mask));
            ARROW_ASSIGN_OR_RAISE(auto new_col, Column::Make(col.name(), std::move(arr)));
            filtered.push_back(std::move(new_col));
        }
        return from_ordered_columns(std::move(filtered));
    };
    return helper().ValueOrDie();
}

EagerDataFrame EagerDataFrame::with_column(const std::string& name, const Expr& expr) const {
    auto helper = [&]() -> arrow::Result<EagerDataFrame> {
        ARROW_ASSIGN_OR_RAISE(auto evaluated, EvaluateExprToColumn(*this, expr, name));
        ARROW_ASSIGN_OR_RAISE(auto new_col, Column::Make(name, evaluated.data()));

        std::vector<Column> updated;
        updated.reserve(columns_.size() + 1);
        bool replaced = false;
        for (const auto& col : columns_) {
            if (col.name() == name) {
                updated.push_back(new_col);
                replaced = true;
            } else {
                updated.push_back(col);
            }
        }
        if (!replaced) {
            updated.push_back(std::move(new_col));
        }
        return from_ordered_columns(std::move(updated));
    };
    return helper().ValueOrDie();
}

EagerDataFrame EagerDataFrame::head(int64_t n) const {
    auto helper = [&]() -> arrow::Result<EagerDataFrame> {
        if (n < 0) {
            return arrow::Status::Invalid("head(n) requires n >= 0.");
        }

        const int64_t limit = std::min(n, num_rows_);
        std::vector<Column> sliced;
        sliced.reserve(columns_.size());
        for (const auto& col : columns_) {
            ARROW_ASSIGN_OR_RAISE(auto new_col, Column::Make(col.name(), col.data()->Slice(0, limit)));
            sliced.push_back(std::move(new_col));
        }
        return from_ordered_columns(std::move(sliced));
    };
    return helper().ValueOrDie();
}

EagerDataFrame EagerDataFrame::sort(
    const std::vector<std::string>& sort_columns,
    const std::vector<bool>& ascending) const {
    auto helper = [&]() -> arrow::Result<EagerDataFrame> {
        if (sort_columns.empty()) {
            return arrow::Status::Invalid("sort() requires at least one key column.");
        }
        if (sort_columns.size() != ascending.size()) {
            return arrow::Status::Invalid(
                "sort(): columns and ascending vectors must have the same length.");
        }

        std::vector<const Column*> key_cols;
        key_cols.reserve(sort_columns.size());
        for (const auto& name : sort_columns) {
            ARROW_ASSIGN_OR_RAISE(const Column* col_ptr, column(name));
            key_cols.push_back(col_ptr);
        }

        std::vector<int64_t> order(static_cast<size_t>(num_rows_));
        for (int64_t i = 0; i < num_rows_; ++i) {
            order[static_cast<size_t>(i)] = i;
        }

        auto compare_rows = [&](int64_t row_a, int64_t row_b) -> bool {
            for (size_t k = 0; k < key_cols.size(); ++k) {
                const auto& arr = key_cols[k]->data();
                const bool asc = ascending[k];

                const bool a_null = arr->IsNull(row_a);
                const bool b_null = arr->IsNull(row_b);

                if (a_null && b_null) continue;
                if (a_null) return false;  
                if (b_null) return true;   

                int cmp = 0;
                switch (arr->type_id()) {
                    case arrow::Type::INT32: {
                        const auto& a = static_cast<const arrow::Int32Array&>(*arr);
                        const int32_t va = a.Value(row_a), vb = a.Value(row_b);
                        cmp = (va < vb) ? -1 : (va > vb) ? 1 : 0;
                        break;
                    }
                    case arrow::Type::INT64: {
                        const auto& a = static_cast<const arrow::Int64Array&>(*arr);
                        const int64_t va = a.Value(row_a), vb = a.Value(row_b);
                        cmp = (va < vb) ? -1 : (va > vb) ? 1 : 0;
                        break;
                    }
                    case arrow::Type::FLOAT: {
                        const auto& a = static_cast<const arrow::FloatArray&>(*arr);
                        const float va = a.Value(row_a), vb = a.Value(row_b);
                        cmp = (va < vb) ? -1 : (va > vb) ? 1 : 0;
                        break;
                    }
                    case arrow::Type::DOUBLE: {
                        const auto& a = static_cast<const arrow::DoubleArray&>(*arr);
                        const double va = a.Value(row_a), vb = a.Value(row_b);
                        cmp = (va < vb) ? -1 : (va > vb) ? 1 : 0;
                        break;
                    }
                    case arrow::Type::STRING: {
                        const auto& a = static_cast<const arrow::StringArray&>(*arr);
                        const std::string va = a.GetString(row_a), vb = a.GetString(row_b);
                        cmp = (va < vb) ? -1 : (va > vb) ? 1 : 0;
                        break;
                    }
                    case arrow::Type::BOOL: {
                        const auto& a = static_cast<const arrow::BooleanArray&>(*arr);
                        const int va = a.Value(row_a) ? 1 : 0;
                        const int vb = a.Value(row_b) ? 1 : 0;
                        cmp = va - vb;
                        break;
                    }
                    default:
                        break;
                }

                if (cmp != 0) {
                    return asc ? (cmp < 0) : (cmp > 0);
                }
            }
            return false;
        };

        std::stable_sort(order.begin(), order.end(), compare_rows);

        std::vector<Column> reindexed;
        reindexed.reserve(columns_.size());
        for (const auto& col : columns_) {
            const auto& arr = col.data();
            std::shared_ptr<arrow::Array> out;
            switch (arr->type_id()) {
                case arrow::Type::INT32: {
                    const auto& src = static_cast<const arrow::Int32Array&>(*arr);
                    arrow::Int32Builder builder;
                    for (int64_t i : order) {
                        ARROW_RETURN_NOT_OK(src.IsNull(i) ? builder.AppendNull() : builder.Append(src.Value(i)));
                    }
                    ARROW_RETURN_NOT_OK(builder.Finish(&out));
                    break;
                }
                case arrow::Type::INT64: {
                    const auto& src = static_cast<const arrow::Int64Array&>(*arr);
                    arrow::Int64Builder builder;
                    for (int64_t i : order) {
                        ARROW_RETURN_NOT_OK(src.IsNull(i) ? builder.AppendNull() : builder.Append(src.Value(i)));
                    }
                    ARROW_RETURN_NOT_OK(builder.Finish(&out));
                    break;
                }
                case arrow::Type::FLOAT: {
                    const auto& src = static_cast<const arrow::FloatArray&>(*arr);
                    arrow::FloatBuilder builder;
                    for (int64_t i : order) {
                        ARROW_RETURN_NOT_OK(src.IsNull(i) ? builder.AppendNull() : builder.Append(src.Value(i)));
                    }
                    ARROW_RETURN_NOT_OK(builder.Finish(&out));
                    break;
                }
                case arrow::Type::DOUBLE: {
                    const auto& src = static_cast<const arrow::DoubleArray&>(*arr);
                    arrow::DoubleBuilder builder;
                    for (int64_t i : order) {
                        ARROW_RETURN_NOT_OK(src.IsNull(i) ? builder.AppendNull() : builder.Append(src.Value(i)));
                    }
                    ARROW_RETURN_NOT_OK(builder.Finish(&out));
                    break;
                }
                case arrow::Type::STRING: {
                    const auto& src = static_cast<const arrow::StringArray&>(*arr);
                    arrow::StringBuilder builder;
                    for (int64_t i : order) {
                        ARROW_RETURN_NOT_OK(src.IsNull(i) ? builder.AppendNull() : builder.Append(src.GetString(i)));
                    }
                    ARROW_RETURN_NOT_OK(builder.Finish(&out));
                    break;
                }
                case arrow::Type::BOOL: {
                    const auto& src = static_cast<const arrow::BooleanArray&>(*arr);
                    arrow::BooleanBuilder builder;
                    for (int64_t i : order) {
                        ARROW_RETURN_NOT_OK(src.IsNull(i) ? builder.AppendNull() : builder.Append(src.Value(i)));
                    }
                    ARROW_RETURN_NOT_OK(builder.Finish(&out));
                    break;
                }
                default:
                    return arrow::Status::TypeError(
                        "Unsupported column type in sort: ", arr->type()->ToString());
            }
            ARROW_ASSIGN_OR_RAISE(auto new_col, Column::Make(col.name(), std::move(out)));
            reindexed.push_back(std::move(new_col));
        }

        return from_ordered_columns(std::move(reindexed));
    };
    return helper().ValueOrDie();
}

GroupedEagerDataFrame EagerDataFrame::group_by(const std::vector<std::string>& keys) const {
    return GroupedEagerDataFrame(*this, keys);
}

EagerDataFrame GroupedEagerDataFrame::aggregate(const std::vector<std::pair<std::string, std::string>>& aggs) const {
    auto helper = [&]() -> arrow::Result<EagerDataFrame> {
        
        std::unordered_map<std::string, Expr> agg_map;
        for (const auto& pair : aggs) {
            const std::string& input_col = pair.first;
            const std::string& op_str = pair.second;
            std::string output_name = input_col + "_" + op_str;
            Expr agg_expr = col(input_col);
            if (op_str == "sum") agg_expr = agg_expr.sum();
            else if (op_str == "mean") agg_expr = agg_expr.mean();
            else if (op_str == "min") agg_expr = agg_expr.min();
            else if (op_str == "max") agg_expr = agg_expr.max();
            else if (op_str == "count") agg_expr = agg_expr.count();
            agg_map.emplace(output_name, agg_expr);
        }

        for (const auto& key : keys_) {
            ARROW_RETURN_NOT_OK(df_.column(key).status());
        }

        std::vector<std::string> group_order;
        std::unordered_map<std::string, std::vector<int64_t>> groups;

        for (int64_t row = 0; row < df_.num_rows(); ++row) {
            std::string key_str;
            for (const auto& kname : keys_) {
                const auto& arr = df_.column(kname).ValueUnsafe()->data();
                key_str += CellToKeyString(arr, row) + "\x00";
            }
            if (groups.find(key_str) == groups.end()) {
                group_order.push_back(key_str);
            }
            groups[key_str].push_back(row);
        }

        std::vector<std::vector<int64_t>> ordered_groups;
        ordered_groups.reserve(group_order.size());
        for (const auto& gk : group_order) {
            ordered_groups.push_back(groups.at(gk));
        }

        std::vector<Column> result;

        std::vector<int64_t> first_rows;
        first_rows.reserve(ordered_groups.size());
        for (const auto& grp : ordered_groups) {
            first_rows.push_back(grp.front());
        }
        for (const auto& kname : keys_) {
            const auto& arr = df_.column(kname).ValueUnsafe()->data();
            ARROW_ASSIGN_OR_RAISE(auto out_arr, GatherArray(arr, first_rows));
            ARROW_ASSIGN_OR_RAISE(auto key_col, Column::Make(kname, std::move(out_arr)));
            result.push_back(std::move(key_col));
        }

        std::vector<std::string> agg_names;
        agg_names.reserve(agg_map.size());
        for (const auto& kv : agg_map) agg_names.push_back(kv.first);
        std::sort(agg_names.begin(), agg_names.end());

        for (const auto& out_name : agg_names) {
            const Expr& agg_expr = agg_map.at(out_name);

            if (agg_expr.node().kind() != ExprNode::Kind::Agg) {
                return arrow::Status::Invalid(
                    "aggregate(): expression for '", out_name,
                    "' is not an aggregation expression (sum/mean/count/min/max).");
            }
            const auto& agg_node = static_cast<const AggNode&>(agg_expr.node());

            ARROW_ASSIGN_OR_RAISE(Column inner_col,
                EvaluateNodeToColumn(df_, agg_node.input(), out_name + "_inner"));

            ARROW_ASSIGN_OR_RAISE(auto out_arr,
                AggregateArrayByGroups(inner_col.data(), ordered_groups, agg_node.op()));

            ARROW_ASSIGN_OR_RAISE(auto agg_col, Column::Make(out_name, std::move(out_arr)));
            result.push_back(std::move(agg_col));
        }

        return EagerDataFrame::from_ordered_columns(std::move(result));
    };
    return helper().ValueOrDie();
}

EagerDataFrame EagerDataFrame::join(
    const EagerDataFrame& other,
    const std::vector<std::string>& on,
    const std::string& how) const {
    auto helper = [&]() -> arrow::Result<EagerDataFrame> {
        if (on.empty()) {
            return arrow::Status::Invalid("join(): 'on' must specify at least one key column.");
        }
        if (how != "inner" && how != "left" && how != "right" && how != "outer") {
            return arrow::Status::Invalid(
                "join(): 'how' must be one of 'inner', 'left', 'right', 'outer'. Got: ", how);
        }

        const std::unordered_set<std::string> key_set(on.begin(), on.end());

        for (const auto& k : on) {
            ARROW_RETURN_NOT_OK(column(k).status());
            ARROW_RETURN_NOT_OK(other.column(k).status());

            const auto& left_type = column(k).ValueUnsafe()->data()->type();
            const auto& right_type = other.column(k).ValueUnsafe()->data()->type();
            if (!left_type->Equals(right_type)) {
                return arrow::Status::TypeError(
                    "join(): key column '", k, "' has mismatched types.");
            }
        }

        std::unordered_set<std::string> left_names;
        for (const auto& c : columns_) left_names.insert(c.name());

        std::vector<std::string> right_extra;
        for (const auto& c : other.columns_) {
            if (key_set.count(c.name())) continue;
            if (left_names.count(c.name())) {
                return arrow::Status::Invalid(
                    "join(): non-key column '", c.name(),
                    "' exists in both DataFrames. Rename one before joining.");
            }
            right_extra.push_back(c.name());
        }

        std::unordered_map<std::string, std::vector<int64_t>> right_index;
        for (int64_t r = 0; r < other.num_rows(); ++r) {
            bool any_null = false;
            std::string ks;
            for (const auto& k : on) {
                const auto& arr = other.column(k).ValueUnsafe()->data();
                if (arr->IsNull(r)) { any_null = true; break; }
                ks += CellToKeyString(arr, r) + "\x00";
            }
            if (!any_null) {
                right_index[ks].push_back(r);
            }
        }

        std::vector<int64_t> left_idx, right_idx;
        std::vector<bool> right_matched(static_cast<size_t>(other.num_rows()), false);

        for (int64_t l = 0; l < num_rows_; ++l) {
            bool any_null = false;
            std::string ks;
            for (const auto& k : on) {
                const auto& arr = column(k).ValueUnsafe()->data();
                if (arr->IsNull(l)) { any_null = true; break; }
                ks += CellToKeyString(arr, l) + "\x00";
            }

            bool matched = false;
            if (!any_null) {
                auto it = right_index.find(ks);
                if (it != right_index.end()) {
                    for (int64_t r : it->second) {
                        left_idx.push_back(l);
                        right_idx.push_back(r);
                        right_matched[static_cast<size_t>(r)] = true;
                        matched = true;
                    }
                }
            }

            if (!matched && (how == "left" || how == "outer")) {
                left_idx.push_back(l);
                right_idx.push_back(-1);
            }
        }

        if (how == "right" || how == "outer") {
            for (int64_t r = 0; r < other.num_rows(); ++r) {
                if (!right_matched[static_cast<size_t>(r)]) {
                    left_idx.push_back(-1);
                    right_idx.push_back(r);
                }
            }
        }

        std::vector<Column> result;

        for (const auto& kname : on) {
            const auto& larr = column(kname).ValueUnsafe()->data();
            const auto& rarr = other.column(kname).ValueUnsafe()->data();
            ARROW_ASSIGN_OR_RAISE(auto lga, GatherArray(larr, left_idx));
            ARROW_ASSIGN_OR_RAISE(auto rga, GatherArray(rarr, right_idx));

            const size_t nr = left_idx.size();
            std::shared_ptr<arrow::Array> merged;

            switch (lga->type_id()) {
                case arrow::Type::INT32: {
                    const auto& la = static_cast<const arrow::Int32Array&>(*lga);
                    const auto& ra = static_cast<const arrow::Int32Array&>(*rga);
                    arrow::Int32Builder b;
                    for (size_t i = 0; i < nr; ++i) {
                        if (!la.IsNull((int64_t)i))      { ARROW_RETURN_NOT_OK(b.Append(la.Value((int64_t)i))); }
                        else if (!ra.IsNull((int64_t)i)) { ARROW_RETURN_NOT_OK(b.Append(ra.Value((int64_t)i))); }
                        else                             { ARROW_RETURN_NOT_OK(b.AppendNull()); }
                    }
                    ARROW_RETURN_NOT_OK(b.Finish(&merged)); break;
                }
                case arrow::Type::INT64: {
                    const auto& la = static_cast<const arrow::Int64Array&>(*lga);
                    const auto& ra = static_cast<const arrow::Int64Array&>(*rga);
                    arrow::Int64Builder b;
                    for (size_t i = 0; i < nr; ++i) {
                        if (!la.IsNull((int64_t)i))      { ARROW_RETURN_NOT_OK(b.Append(la.Value((int64_t)i))); }
                        else if (!ra.IsNull((int64_t)i)) { ARROW_RETURN_NOT_OK(b.Append(ra.Value((int64_t)i))); }
                        else                             { ARROW_RETURN_NOT_OK(b.AppendNull()); }
                    }
                    ARROW_RETURN_NOT_OK(b.Finish(&merged)); break;
                }
                case arrow::Type::FLOAT: {
                    const auto& la = static_cast<const arrow::FloatArray&>(*lga);
                    const auto& ra = static_cast<const arrow::FloatArray&>(*rga);
                    arrow::FloatBuilder b;
                    for (size_t i = 0; i < nr; ++i) {
                        if (!la.IsNull((int64_t)i))      { ARROW_RETURN_NOT_OK(b.Append(la.Value((int64_t)i))); }
                        else if (!ra.IsNull((int64_t)i)) { ARROW_RETURN_NOT_OK(b.Append(ra.Value((int64_t)i))); }
                        else                             { ARROW_RETURN_NOT_OK(b.AppendNull()); }
                    }
                    ARROW_RETURN_NOT_OK(b.Finish(&merged)); break;
                }
                case arrow::Type::DOUBLE: {
                    const auto& la = static_cast<const arrow::DoubleArray&>(*lga);
                    const auto& ra = static_cast<const arrow::DoubleArray&>(*rga);
                    arrow::DoubleBuilder b;
                    for (size_t i = 0; i < nr; ++i) {
                        if (!la.IsNull((int64_t)i))      { ARROW_RETURN_NOT_OK(b.Append(la.Value((int64_t)i))); }
                        else if (!ra.IsNull((int64_t)i)) { ARROW_RETURN_NOT_OK(b.Append(ra.Value((int64_t)i))); }
                        else                             { ARROW_RETURN_NOT_OK(b.AppendNull()); }
                    }
                    ARROW_RETURN_NOT_OK(b.Finish(&merged)); break;
                }
                case arrow::Type::BOOL: {
                    const auto& la = static_cast<const arrow::BooleanArray&>(*lga);
                    const auto& ra = static_cast<const arrow::BooleanArray&>(*rga);
                    arrow::BooleanBuilder b;
                    for (size_t i = 0; i < nr; ++i) {
                        if (!la.IsNull((int64_t)i))      { ARROW_RETURN_NOT_OK(b.Append(la.Value((int64_t)i))); }
                        else if (!ra.IsNull((int64_t)i)) { ARROW_RETURN_NOT_OK(b.Append(ra.Value((int64_t)i))); }
                        else                             { ARROW_RETURN_NOT_OK(b.AppendNull()); }
                    }
                    ARROW_RETURN_NOT_OK(b.Finish(&merged)); break;
                }
                case arrow::Type::STRING: {
                    const auto& la = static_cast<const arrow::StringArray&>(*lga);
                    const auto& ra = static_cast<const arrow::StringArray&>(*rga);
                    arrow::StringBuilder b;
                    for (size_t i = 0; i < nr; ++i) {
                        if (!la.IsNull((int64_t)i))      { ARROW_RETURN_NOT_OK(b.Append(la.GetString((int64_t)i))); }
                        else if (!ra.IsNull((int64_t)i)) { ARROW_RETURN_NOT_OK(b.Append(ra.GetString((int64_t)i))); }
                        else                             { ARROW_RETURN_NOT_OK(b.AppendNull()); }
                    }
                    ARROW_RETURN_NOT_OK(b.Finish(&merged)); break;
                }
                default:
                    return arrow::Status::TypeError("join: unsupported key column type: ", lga->type()->ToString());
            }
            ARROW_ASSIGN_OR_RAISE(auto key_col, Column::Make(kname, std::move(merged)));
            result.push_back(std::move(key_col));
        }

        for (const auto& c : columns_) {
            if (key_set.count(c.name())) continue;
            ARROW_ASSIGN_OR_RAISE(auto col_arr, GatherArray(c.data(), left_idx));
            ARROW_ASSIGN_OR_RAISE(auto left_col, Column::Make(c.name(), std::move(col_arr)));
            result.push_back(std::move(left_col));
        }

        for (const auto& name : right_extra) {
            const auto& arr = other.column(name).ValueUnsafe()->data();
            ARROW_ASSIGN_OR_RAISE(auto col_arr, GatherArray(arr, right_idx));
            ARROW_ASSIGN_OR_RAISE(auto right_col, Column::Make(name, std::move(col_arr)));
            result.push_back(std::move(right_col));
        }

        return EagerDataFrame::from_ordered_columns(std::move(result));
    };
    return helper().ValueOrDie();
}

}  