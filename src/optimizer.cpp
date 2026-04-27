#include "dataframelib/optimizer.hpp"
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <arrow/scalar.h>
#include "dataframelib/expr.hpp"
namespace dataframelib {
namespace {
struct LiteralValue {
    enum class Kind {
        kInt32,
        kInt64,
        kFloat,
        kDouble,
        kBool,
        kString,
    };
    Kind kind;
    int32_t i32=0;
    int64_t i64=0;
    float f32=0.0f;
    double f64=0.0;
    bool b=false;
    std::string s;
};

bool TryGetLiteralValue(const Expr& expr, LiteralValue* out) {
    if (expr.node().kind()!=ExprNode::Kind::Lit) {
        return false;
    }
    const auto& lit_node=static_cast<const LitNode&>(expr.node());
    const auto& scalar=lit_node.scalar();
    if (!scalar||!scalar->is_valid) {
        return false;
    }
    switch (scalar->type->id()) {
        case arrow::Type::INT32: {
            out->kind=LiteralValue::Kind::kInt32;
            out->i32=static_cast<const arrow::Int32Scalar&>(*scalar).value;
            return true;
        }
        case arrow::Type::INT64: {
            out->kind=LiteralValue::Kind::kInt64;
            out->i64=static_cast<const arrow::Int64Scalar&>(*scalar).value;
            return true;
        }
        case arrow::Type::FLOAT: {
            out->kind=LiteralValue::Kind::kFloat;
            out->f32=static_cast<const arrow::FloatScalar&>(*scalar).value;
            return true;
        }
        case arrow::Type::DOUBLE: {
            out->kind=LiteralValue::Kind::kDouble;
            out->f64=static_cast<const arrow::DoubleScalar&>(*scalar).value;
            return true;
        }
        case arrow::Type::BOOL: {
            out->kind=LiteralValue::Kind::kBool;
            out->b=static_cast<const arrow::BooleanScalar&>(*scalar).value;
            return true;
        }
        case arrow::Type::STRING: {
            out->kind=LiteralValue::Kind::kString;
            out->s=static_cast<const arrow::StringScalar&>(*scalar).ToString();
            return true;
        }
        default:
            return false;
    }
}

bool IsNumeric(const LiteralValue& v) {
    return v.kind==LiteralValue::Kind::kInt32 ||
           v.kind==LiteralValue::Kind::kInt64 ||
           v.kind==LiteralValue::Kind::kFloat ||
           v.kind==LiteralValue::Kind::kDouble;
}

bool IsInteger(const LiteralValue& v) {
    return v.kind==LiteralValue::Kind::kInt32||v.kind==LiteralValue::Kind::kInt64;
}

bool IsZero(const LiteralValue& v) {
    switch (v.kind) {
        case LiteralValue::Kind::kInt32:
            return v.i32==0;
        case LiteralValue::Kind::kInt64:
            return v.i64==0;
        case LiteralValue::Kind::kFloat:
            return v.f32==0.0f;
        case LiteralValue::Kind::kDouble:
            return v.f64==0.0;
        default:
            return false;
    }
}

bool IsOne(const LiteralValue& v) {
    switch (v.kind) {
        case LiteralValue::Kind::kInt32:
            return v.i32==1;
        case LiteralValue::Kind::kInt64:
            return v.i64==1;
        case LiteralValue::Kind::kFloat:
            return v.f32==1.0f;
        case LiteralValue::Kind::kDouble:
            return v.f64==1.0;
        default:
            return false;
    }
}

bool IsBoolLiteral(const Expr& expr, bool value) {
    LiteralValue lit;
    if (!TryGetLiteralValue(expr, &lit)) {
        return false;
    }
    return lit.kind==LiteralValue::Kind::kBool&&lit.b==value;
}

std::optional<Expr> TryBuildLiteralExpr(const LiteralValue& value) {
    switch (value.kind) {
        case LiteralValue::Kind::kInt32:
            return lit(value.i32);
        case LiteralValue::Kind::kInt64:
            return lit(value.i64);
        case LiteralValue::Kind::kFloat:
            return lit(value.f32);
        case LiteralValue::Kind::kDouble:
            return lit(value.f64);
        case LiteralValue::Kind::kBool:
            return lit(value.b);
        case LiteralValue::Kind::kString:
            return lit(value.s);
    }
    return std::nullopt;
}

std::optional<LiteralValue> FoldBinaryLiteral(BinaryOpKind op,
                                              const LiteralValue& left,
                                              const LiteralValue& right) {
    LiteralValue out;
    if (IsNumeric(left)&&IsNumeric(right)) {
        const bool promote_double=left.kind==LiteralValue::Kind::kDouble ||
                                    right.kind==LiteralValue::Kind::kDouble;
        const bool promote_float=!promote_double &&
                                   (left.kind==LiteralValue::Kind::kFloat ||
                                    right.kind==LiteralValue::Kind::kFloat);
        const bool promote_int64=!promote_double&&!promote_float &&
                                   (left.kind==LiteralValue::Kind::kInt64 ||
                                    right.kind==LiteralValue::Kind::kInt64);
        const auto as_double=[](const LiteralValue& v) {
            if (v.kind==LiteralValue::Kind::kInt32) return static_cast<double>(v.i32);
            if (v.kind==LiteralValue::Kind::kInt64) return static_cast<double>(v.i64);
            if (v.kind==LiteralValue::Kind::kFloat) return static_cast<double>(v.f32);
            return v.f64;
        };
        const auto as_int64=[](const LiteralValue& v) {
            return v.kind==LiteralValue::Kind::kInt32 ? static_cast<int64_t>(v.i32) : v.i64;
        };
        if (op==BinaryOpKind::kAdd||op==BinaryOpKind::kSub ||
            op==BinaryOpKind::kMul||op==BinaryOpKind::kDiv) {
            if (promote_double||promote_float) {
                const double l=as_double(left);
                const double r=as_double(right);
                if (op==BinaryOpKind::kDiv&&r==0.0) {
                    return std::nullopt;
                }
                double v=0.0;
                if (op==BinaryOpKind::kAdd) v=l+r;
                if (op==BinaryOpKind::kSub) v=l-r;
                if (op==BinaryOpKind::kMul) v=l*r;
                if (op==BinaryOpKind::kDiv) v=l/r;
                if (promote_double) {
                    out.kind=LiteralValue::Kind::kDouble;
                    out.f64=v;
                } else {
                    out.kind=LiteralValue::Kind::kFloat;
                    out.f32=static_cast<float>(v);
                }
                return out;
            }
            const int64_t l=as_int64(left);
            const int64_t r=as_int64(right);
            if (op==BinaryOpKind::kDiv&&r==0) {
                return std::nullopt;
            }
            int64_t v=0;
            if (op==BinaryOpKind::kAdd) v=l+r;
            if (op==BinaryOpKind::kSub) v=l-r;
            if (op==BinaryOpKind::kMul) v=l*r;
            if (op==BinaryOpKind::kDiv) v=l/r;
            if (promote_int64) {
                out.kind=LiteralValue::Kind::kInt64;
                out.i64=v;
            } else {
                out.kind=LiteralValue::Kind::kInt32;
                out.i32=static_cast<int32_t>(v);
            }
            return out;
        }
        if (op==BinaryOpKind::kMod) {
            if (!IsInteger(left)||!IsInteger(right)) {
                return std::nullopt;
            }
            const int64_t l=as_int64(left);
            const int64_t r=as_int64(right);
            if (r==0) {
                return std::nullopt;
            }
            const int64_t v=l%r;
            if (promote_int64) {
                out.kind=LiteralValue::Kind::kInt64;
                out.i64=v;
            } else {
                out.kind=LiteralValue::Kind::kInt32;
                out.i32=static_cast<int32_t>(v);
            }
            return out;
        }
        if (op==BinaryOpKind::kEq||op==BinaryOpKind::kNeq ||
            op==BinaryOpKind::kLt||op==BinaryOpKind::kLe ||
            op==BinaryOpKind::kGt||op==BinaryOpKind::kGe) {
            const double l=as_double(left);
            const double r=as_double(right);
            bool cmp=false;
            if (op==BinaryOpKind::kEq) cmp=(l==r);
            if (op==BinaryOpKind::kNeq) cmp=(l!=r);
            if (op==BinaryOpKind::kLt) cmp=(l<r);
            if (op==BinaryOpKind::kLe) cmp=(l<=r);
            if (op==BinaryOpKind::kGt) cmp=(l>r);
            if (op==BinaryOpKind::kGe) cmp=(l>=r);
            out.kind=LiteralValue::Kind::kBool;
            out.b=cmp;
            return out;
        }
    }
    if (left.kind==LiteralValue::Kind::kBool&&right.kind==LiteralValue::Kind::kBool) {
        if (op==BinaryOpKind::kAnd||op==BinaryOpKind::kOr ||
            op==BinaryOpKind::kEq||op==BinaryOpKind::kNeq) {
            out.kind=LiteralValue::Kind::kBool;
            if (op==BinaryOpKind::kAnd) out.b=left.b&&right.b;
            if (op==BinaryOpKind::kOr) out.b=left.b||right.b;
            if (op==BinaryOpKind::kEq) out.b=(left.b==right.b);
            if (op==BinaryOpKind::kNeq) out.b=(left.b!=right.b);
            return out;
        }
    }
    if (left.kind==LiteralValue::Kind::kString&&right.kind==LiteralValue::Kind::kString) {
        if (op==BinaryOpKind::kEq||op==BinaryOpKind::kNeq ||
            op==BinaryOpKind::kLt||op==BinaryOpKind::kLe ||
            op==BinaryOpKind::kGt||op==BinaryOpKind::kGe) {
            out.kind=LiteralValue::Kind::kBool;
            if (op==BinaryOpKind::kEq) out.b=(left.s==right.s);
            if (op==BinaryOpKind::kNeq) out.b=(left.s!=right.s);
            if (op==BinaryOpKind::kLt) out.b=(left.s<right.s);
            if (op==BinaryOpKind::kLe) out.b=(left.s<=right.s);
            if (op==BinaryOpKind::kGt) out.b=(left.s>right.s);
            if (op==BinaryOpKind::kGe) out.b=(left.s>=right.s);
            return out;
        }
    }
    return std::nullopt;
}

std::optional<LiteralValue> FoldUnaryLiteral(UnaryOpKind op, const LiteralValue& input) {
    LiteralValue out;
    if (op==UnaryOpKind::kNot&&input.kind==LiteralValue::Kind::kBool) {
        out.kind=LiteralValue::Kind::kBool;
        out.b=!input.b;
        return out;
    }
    if (op==UnaryOpKind::kAbs&&IsNumeric(input)) {
        if (input.kind==LiteralValue::Kind::kInt32) {
            out.kind=LiteralValue::Kind::kInt32;
            out.i32=input.i32<0 ? -input.i32 : input.i32;
            return out;
        }
        if (input.kind==LiteralValue::Kind::kInt64) {
            out.kind=LiteralValue::Kind::kInt64;
            out.i64=input.i64<0 ? -input.i64 : input.i64;
            return out;
        }
        if (input.kind==LiteralValue::Kind::kFloat) {
            out.kind=LiteralValue::Kind::kFloat;
            out.f32=input.f32<0.0f ? -input.f32 : input.f32;
            return out;
        }
        out.kind=LiteralValue::Kind::kDouble;
        out.f64=input.f64<0.0 ? -input.f64 : input.f64;
        return out;
    }
    if (op==UnaryOpKind::kIsNull||op==UnaryOpKind::kIsNotNull) {
        out.kind=LiteralValue::Kind::kBool;
        out.b=(op==UnaryOpKind::kIsNotNull);
        return out;
    }
    return std::nullopt;
}

std::optional<LiteralValue> FoldStringLiteral(StringOpKind op,
                                              const LiteralValue& input,
                                              const std::string& arg) {
    if (input.kind!=LiteralValue::Kind::kString) {
        return std::nullopt;
    }
    LiteralValue out;
    if (op==StringOpKind::kLength) {
        out.kind=LiteralValue::Kind::kInt32;
        out.i32=static_cast<int32_t>(input.s.size());
        return out;
    }
    if (op==StringOpKind::kContains) {
        out.kind=LiteralValue::Kind::kBool;
        out.b=input.s.find(arg)!=std::string::npos;
        return out;
    }
    if (op==StringOpKind::kStartsWith) {
        out.kind=LiteralValue::Kind::kBool;
        out.b=input.s.rfind(arg, 0)==0;
        return out;
    }
    if (op==StringOpKind::kEndsWith) {
        out.kind=LiteralValue::Kind::kBool;
        out.b=input.s.size()>=arg.size() &&
                input.s.compare(input.s.size()-arg.size(), arg.size(), arg)==0;
        return out;
    }
    if (op==StringOpKind::kToLower) {
        out.kind=LiteralValue::Kind::kString;
        out.s=input.s;
        std::transform(out.s.begin(), out.s.end(), out.s.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return out;
    }
    if (op==StringOpKind::kToUpper) {
        out.kind=LiteralValue::Kind::kString;
        out.s=input.s;
        std::transform(out.s.begin(), out.s.end(), out.s.begin(), [](unsigned char c) {
            return static_cast<char>(std::toupper(c));
        });
        return out;
    }
    return std::nullopt;
}

std::optional<Expr> TryExprFromNode(const std::shared_ptr<ExprNode>& node);
Expr OptimizeExpr(const Expr& expr);
Expr OptimizeBinaryExpr(const Expr& original, const BinaryNode& node) {
    const auto left_base=TryExprFromNode(node.left());
    const auto right_base=TryExprFromNode(node.right());
    if (!left_base||!right_base) {
        return original;
    }
    Expr left=OptimizeExpr(*left_base);
    Expr right=OptimizeExpr(*right_base);
    LiteralValue left_lit;
    LiteralValue right_lit;
    if (TryGetLiteralValue(left, &left_lit)&&TryGetLiteralValue(right, &right_lit)) {
        auto folded=FoldBinaryLiteral(node.op(), left_lit, right_lit);
        if (folded.has_value()) {
            auto lit_expr=TryBuildLiteralExpr(*folded);
            if (lit_expr.has_value()) {
                return *lit_expr;
            }
        }
    }
    if (node.op()==BinaryOpKind::kAdd) {
        if (TryGetLiteralValue(right, &right_lit)&&IsZero(right_lit)) {
            return left;
        }
        if (TryGetLiteralValue(left, &left_lit)&&IsZero(left_lit)) {
            return right;
        }
    }
    if (node.op()==BinaryOpKind::kSub) {
        if (TryGetLiteralValue(right, &right_lit)&&IsZero(right_lit)) {
            return left;
        }
    }
    if (node.op()==BinaryOpKind::kMul) {
        if (TryGetLiteralValue(right, &right_lit)&&IsOne(right_lit)) {
            return left;
        }
        if (TryGetLiteralValue(left, &left_lit)&&IsOne(left_lit)) {
            return right;
        }
    }
    if (node.op()==BinaryOpKind::kDiv) {
        if (TryGetLiteralValue(right, &right_lit)&&IsOne(right_lit)) {
            return left;
        }
    }
    if (node.op()==BinaryOpKind::kAnd) {
        if (IsBoolLiteral(left, true)) {
            return right;
        }
        if (IsBoolLiteral(right, true)) {
            return left;
        }
    }
    if (node.op()==BinaryOpKind::kOr) {
        if (IsBoolLiteral(left, false)) {
            return right;
        }
        if (IsBoolLiteral(right, false)) {
            return left;
        }
    }
    switch (node.op()) {
        case BinaryOpKind::kAdd:
            return left+right;
        case BinaryOpKind::kSub:
            return left-right;
        case BinaryOpKind::kMul:
            return left*right;
        case BinaryOpKind::kDiv:
            return left/right;
        case BinaryOpKind::kMod:
            return left%right;
        case BinaryOpKind::kEq:
            return left==right;
        case BinaryOpKind::kNeq:
            return left!=right;
        case BinaryOpKind::kLt:
            return left<right;
        case BinaryOpKind::kLe:
            return left<=right;
        case BinaryOpKind::kGt:
            return left>right;
        case BinaryOpKind::kGe:
            return left>=right;
        case BinaryOpKind::kAnd:
            return left&right;
        case BinaryOpKind::kOr:
            return left|right;
    }
    return original;
}

Expr OptimizeUnaryExpr(const Expr& original, const UnaryNode& node) {
    const auto child_base=TryExprFromNode(node.input());
    if (!child_base) {
        return original;
    }
    Expr input=OptimizeExpr(*child_base);
    LiteralValue lit_value;
    if (TryGetLiteralValue(input, &lit_value)) {
        auto folded=FoldUnaryLiteral(node.op(), lit_value);
        if (folded.has_value()) {
            auto lit_expr=TryBuildLiteralExpr(*folded);
            if (lit_expr.has_value()) {
                return *lit_expr;
            }
        }
    }
    if (node.op()==UnaryOpKind::kNot&&input.node().kind()==ExprNode::Kind::Unary) {
        const auto& inner=static_cast<const UnaryNode&>(input.node());
        if (inner.op()==UnaryOpKind::kNot) {
            auto inner_expr=TryExprFromNode(inner.input());
            if (inner_expr.has_value()) {
                return OptimizeExpr(*inner_expr);
            }
        }
    }
    switch (node.op()) {
        case UnaryOpKind::kNot:
            return ~input;
        case UnaryOpKind::kAbs:
            return input.abs();
        case UnaryOpKind::kIsNull:
            return input.is_null();
        case UnaryOpKind::kIsNotNull:
            return input.is_not_null();
    }
    return input;
}

Expr OptimizeStringExpr(const Expr& original, const StringOpNode& node) {
    const auto child_base=TryExprFromNode(node.input());
    if (!child_base) {
        return original;
    }
    Expr input=OptimizeExpr(*child_base);
    LiteralValue lit_value;
    if (TryGetLiteralValue(input, &lit_value)) {
        auto folded=FoldStringLiteral(node.op(), lit_value, node.arg());
        if (folded.has_value()) {
            auto lit_expr=TryBuildLiteralExpr(*folded);
            if (lit_expr.has_value()) {
                return *lit_expr;
            }
        }
    }
    switch (node.op()) {
        case StringOpKind::kLength:
            return input.length();
        case StringOpKind::kContains:
            return input.contains(node.arg());
        case StringOpKind::kStartsWith:
            return input.starts_with(node.arg());
        case StringOpKind::kEndsWith:
            return input.ends_with(node.arg());
        case StringOpKind::kToLower:
            return input.to_lower();
        case StringOpKind::kToUpper:
            return input.to_upper();
    }
    return input;
}

Expr OptimizeAggExpr(const Expr& original, const AggNode& node) {
    const auto child_base=TryExprFromNode(node.input());
    if (!child_base) {
        return original;
    }
    Expr input=OptimizeExpr(*child_base);
    switch (node.op()) {
        case AggOpKind::kSum:
            return input.sum();
        case AggOpKind::kMean:
            return input.mean();
        case AggOpKind::kCount:
            return input.count();
        case AggOpKind::kMin:
            return input.min();
        case AggOpKind::kMax:
            return input.max();
    }
    return input;
}

Expr OptimizeExpr(const Expr& expr) {
    switch (expr.node().kind()) {
        case ExprNode::Kind::Col:
        case ExprNode::Kind::Lit:
            return expr;
        case ExprNode::Kind::Alias: {
            const auto& alias=static_cast<const AliasNode&>(expr.node());
            const auto input_expr=TryExprFromNode(alias.input());
            if (!input_expr.has_value()) {
                return expr;
            }
            return OptimizeExpr(*input_expr).alias(alias.name());
        }
        case ExprNode::Kind::Binary:
            return OptimizeBinaryExpr(expr, static_cast<const BinaryNode&>(expr.node()));
        case ExprNode::Kind::Unary:
            return OptimizeUnaryExpr(expr, static_cast<const UnaryNode&>(expr.node()));
        case ExprNode::Kind::StringOp:
            return OptimizeStringExpr(expr, static_cast<const StringOpNode&>(expr.node()));
        case ExprNode::Kind::Agg:
            return OptimizeAggExpr(expr, static_cast<const AggNode&>(expr.node()));
    }
    return expr;
}

std::optional<Expr> TryExprFromNode(const std::shared_ptr<ExprNode>& node) {
    if (!node) {
        return std::nullopt;
    }
    switch (node->kind()) {
        case ExprNode::Kind::Col: {
            const auto* n=dynamic_cast<const ColNode*>(node.get());
            if (!n) return std::nullopt;
            return col(n->name());
        }
        case ExprNode::Kind::Lit: {
            const auto* n=dynamic_cast<const LitNode*>(node.get());
            if (!n||!n->scalar()||!n->scalar()->is_valid) {
                return std::nullopt;
            }
            switch (n->scalar()->type->id()) {
                case arrow::Type::INT32:
                    return lit(static_cast<const arrow::Int32Scalar&>(*n->scalar()).value);
                case arrow::Type::INT64:
                    return lit(static_cast<const arrow::Int64Scalar&>(*n->scalar()).value);
                case arrow::Type::FLOAT:
                    return lit(static_cast<const arrow::FloatScalar&>(*n->scalar()).value);
                case arrow::Type::DOUBLE:
                    return lit(static_cast<const arrow::DoubleScalar&>(*n->scalar()).value);
                case arrow::Type::BOOL:
                    return lit(static_cast<const arrow::BooleanScalar&>(*n->scalar()).value);
                case arrow::Type::STRING: {
                    const auto& s=static_cast<const arrow::StringScalar&>(*n->scalar());
                    return lit(std::string(reinterpret_cast<const char*>(s.value->data()), s.value->size()));
                }
                default:
                    return std::nullopt;
            }
        }
        case ExprNode::Kind::Alias: {
            const auto* n=dynamic_cast<const AliasNode*>(node.get());
            if (!n) return std::nullopt;
            auto input=TryExprFromNode(n->input());
            if (!input.has_value()) return std::nullopt;
            return input->alias(n->name());
        }
        case ExprNode::Kind::Binary: {
            const auto* n=dynamic_cast<const BinaryNode*>(node.get());
            if (!n) return std::nullopt;
            auto left=TryExprFromNode(n->left());
            auto right=TryExprFromNode(n->right());
            if (!left.has_value()||!right.has_value()) return std::nullopt;
            switch (n->op()) {
                case BinaryOpKind::kAdd:
                    return *left+*right;
                case BinaryOpKind::kSub:
                    return *left-*right;
                case BinaryOpKind::kMul:
                    return *left**right;
                case BinaryOpKind::kDiv:
                    return *left/(*right);
                case BinaryOpKind::kMod:
                    return *left%*right;
                case BinaryOpKind::kEq:
                    return *left==*right;
                case BinaryOpKind::kNeq:
                    return *left!=*right;
                case BinaryOpKind::kLt:
                    return *left<*right;
                case BinaryOpKind::kLe:
                    return *left<=*right;
                case BinaryOpKind::kGt:
                    return *left>*right;
                case BinaryOpKind::kGe:
                    return *left>=*right;
                case BinaryOpKind::kAnd:
                    return *left&*right;
                case BinaryOpKind::kOr:
                    return *left|*right;
            }
            return std::nullopt;
        }
        case ExprNode::Kind::Unary: {
            const auto* n=dynamic_cast<const UnaryNode*>(node.get());
            if (!n) return std::nullopt;
            auto input=TryExprFromNode(n->input());
            if (!input.has_value()) return std::nullopt;
            switch (n->op()) {
                case UnaryOpKind::kNot:
                    return ~(*input);
                case UnaryOpKind::kAbs:
                    return input->abs();
                case UnaryOpKind::kIsNull:
                    return input->is_null();
                case UnaryOpKind::kIsNotNull:
                    return input->is_not_null();
            }
            return std::nullopt;
        }
        case ExprNode::Kind::StringOp: {
            const auto* n=dynamic_cast<const StringOpNode*>(node.get());
            if (!n) return std::nullopt;
            auto input=TryExprFromNode(n->input());
            if (!input.has_value()) return std::nullopt;
            switch (n->op()) {
                case StringOpKind::kLength:
                    return input->length();
                case StringOpKind::kContains:
                    return input->contains(n->arg());
                case StringOpKind::kStartsWith:
                    return input->starts_with(n->arg());
                case StringOpKind::kEndsWith:
                    return input->ends_with(n->arg());
                case StringOpKind::kToLower:
                    return input->to_lower();
                case StringOpKind::kToUpper:
                    return input->to_upper();
            }
            return std::nullopt;
        }
        case ExprNode::Kind::Agg: {
            const auto* n=dynamic_cast<const AggNode*>(node.get());
            if (!n) return std::nullopt;
            auto input=TryExprFromNode(n->input());
            if (!input.has_value()) return std::nullopt;
            switch (n->op()) {
                case AggOpKind::kSum:
                    return input->sum();
                case AggOpKind::kMean:
                    return input->mean();
                case AggOpKind::kCount:
                    return input->count();
                case AggOpKind::kMin:
                    return input->min();
                case AggOpKind::kMax:
                    return input->max();
            }
            return std::nullopt;
        }
    }
    return std::nullopt;
}

void CollectColumnsFromExpr(const std::shared_ptr<ExprNode>& node,
                            std::unordered_set<std::string>* out) {
    if (!node) {
        return;
    }
    switch (node->kind()) {
        case ExprNode::Kind::Col: {
            const auto* col_node=dynamic_cast<const ColNode*>(node.get());
            if (col_node) {
                out->insert(col_node->name());
            }
            return;
        }
        case ExprNode::Kind::Lit:
            return;
        case ExprNode::Kind::Alias: {
            const auto* alias_node=dynamic_cast<const AliasNode*>(node.get());
            if (alias_node) {
                CollectColumnsFromExpr(alias_node->input(), out);
            }
            return;
        }
        case ExprNode::Kind::Binary: {
            const auto* binary_node=dynamic_cast<const BinaryNode*>(node.get());
            if (binary_node) {
                CollectColumnsFromExpr(binary_node->left(), out);
                CollectColumnsFromExpr(binary_node->right(), out);
            }
            return;
        }
        case ExprNode::Kind::Unary: {
            const auto* unary_node=dynamic_cast<const UnaryNode*>(node.get());
            if (unary_node) {
                CollectColumnsFromExpr(unary_node->input(), out);
            }
            return;
        }
        case ExprNode::Kind::StringOp: {
            const auto* string_node=dynamic_cast<const StringOpNode*>(node.get());
            if (string_node) {
                CollectColumnsFromExpr(string_node->input(), out);
            }
            return;
        }
        case ExprNode::Kind::Agg: {
            const auto* agg_node=dynamic_cast<const AggNode*>(node.get());
            if (agg_node) {
                CollectColumnsFromExpr(agg_node->input(), out);
            }
            return;
        }
    }
}

std::vector<std::string> SortedColumns(const std::unordered_set<std::string>& set_like) {
    std::vector<std::string> out(set_like.begin(), set_like.end());
    std::sort(out.begin(), out.end());
    return out;
}

bool IsExactSelectNames(const std::shared_ptr<PlanNode>& node,
                        const std::vector<std::string>& columns) {
    if (!node||node->kind()!=PlanNode::Kind::SelectNames) {
        return false;
    }
    const auto* select=dynamic_cast<const SelectNamesNode*>(node.get());
    if (!select) {
        return false;
    }
    return select->columns()==columns;
}

std::shared_ptr<PlanNode> MakePlanWithOptimizedExprs(const std::shared_ptr<PlanNode>& node);
std::shared_ptr<PlanNode> ApplyPredicatePushdown(const std::shared_ptr<PlanNode>& node) {
    if (!node||node->kind()!=PlanNode::Kind::Filter) {
        return node;
    }
    const auto* outer_filter=dynamic_cast<const FilterNode*>(node.get());
    if (!outer_filter||!outer_filter->input()) {
        return node;
    }
    if (outer_filter->input()->kind()==PlanNode::Kind::SelectNames) {
        const auto* select=dynamic_cast<const SelectNamesNode*>(outer_filter->input().get());
        if (select) {
            std::unordered_set<std::string> predicate_cols;
            CollectColumnsFromExpr(outer_filter->predicate().node_ptr(), &predicate_cols);
            bool subset=true;
            for (const auto& col_name : predicate_cols) {
                if (std::find(select->columns().begin(), select->columns().end(), col_name) ==
                    select->columns().end()) {
                    subset=false;
                    break;
                }
            }
            if (subset) {
                auto pushed_filter=std::make_shared<FilterNode>(select->input(), outer_filter->predicate());
                return std::make_shared<SelectNamesNode>(pushed_filter, select->columns());
            }
        }
    }
    if (outer_filter->input()->kind()==PlanNode::Kind::WithColumn) {
        const auto* with_col=dynamic_cast<const WithColumnNode*>(outer_filter->input().get());
        if (with_col) {
            std::unordered_set<std::string> predicate_cols;
            CollectColumnsFromExpr(outer_filter->predicate().node_ptr(), &predicate_cols);
            if (predicate_cols.find(with_col->name())==predicate_cols.end()) {
                auto pushed_filter =
                    std::make_shared<FilterNode>(with_col->input(), outer_filter->predicate());
                return std::make_shared<WithColumnNode>(pushed_filter, with_col->name(), with_col->expr());
            }
        }
    }
    if (outer_filter->input()->kind()==PlanNode::Kind::Filter) {
        const auto* inner_filter=dynamic_cast<const FilterNode*>(outer_filter->input().get());
        if (inner_filter) {
            Expr combined=OptimizeExpr(inner_filter->predicate()&outer_filter->predicate());
            return std::make_shared<FilterNode>(inner_filter->input(), combined);
        }
    }
    if (outer_filter->input()->kind()==PlanNode::Kind::Sort) {
        const auto* sort=dynamic_cast<const SortNode*>(outer_filter->input().get());
        if (sort) {
            auto pushed_filter =
                std::make_shared<FilterNode>(sort->input(), outer_filter->predicate());
            return std::make_shared<SortNode>(
                pushed_filter, sort->columns(), sort->ascending());
        }
    }
    if (outer_filter->input()->kind()==PlanNode::Kind::Join) {
        const auto* join=dynamic_cast<const JoinNode*>(outer_filter->input().get());
        if (join) {
            std::unordered_set<std::string> predicate_cols;
            CollectColumnsFromExpr(outer_filter->predicate().node_ptr(), &predicate_cols);
            // Check if the filter ONLY uses the Join Keys.
            bool only_uses_join_keys=true;
            for (const auto& col_name : predicate_cols) {
                if (std::find(join->on().begin(), join->on().end(), col_name)==join->on().end()) {
                    only_uses_join_keys=false;
                    break;
                }
            }
            if (only_uses_join_keys) {
                std::shared_ptr<PlanNode> new_left=join->input();
                std::shared_ptr<PlanNode> new_right=join->right_input();
                if (join->how()=="inner") {
                    new_left=std::make_shared<FilterNode>(join->input(), outer_filter->predicate());
                    new_right=std::make_shared<FilterNode>(join->right_input(), outer_filter->predicate());
                } else if (join->how()=="left") {
                    new_left=std::make_shared<FilterNode>(join->input(), outer_filter->predicate());
                } else if (join->how()=="right") {
                    new_right=std::make_shared<FilterNode>(join->right_input(), outer_filter->predicate());
                }
                if (join->how()!="outer") {
                    return std::make_shared<JoinNode>(new_left, new_right, join->on(), join->how());
                }
            }
        }
    }
    return node;
}

std::shared_ptr<PlanNode> ApplyProjectionPushdown(const std::shared_ptr<PlanNode>& node) {
    if (!node) {
        return node;
    }
    if (node->kind()==PlanNode::Kind::SelectNames) {
        const auto* select=dynamic_cast<const SelectNamesNode*>(node.get());
        if (!select||!select->input()) {
            return node;
        }
        if (select->input()->kind()==PlanNode::Kind::Filter) {
            const auto* filter=dynamic_cast<const FilterNode*>(select->input().get());
            if (!filter) {
                return node;
            }
            std::unordered_set<std::string> needed(select->columns().begin(), select->columns().end());
            CollectColumnsFromExpr(filter->predicate().node_ptr(), &needed);
            const auto needed_cols=SortedColumns(needed);
            std::shared_ptr<PlanNode> filtered_input=filter->input();
            if (!IsExactSelectNames(filtered_input, needed_cols)) {
                filtered_input=std::make_shared<SelectNamesNode>(filtered_input, needed_cols);
            }
            auto pushed_filter=std::make_shared<FilterNode>(filtered_input, filter->predicate());
            return std::make_shared<SelectNamesNode>(pushed_filter, select->columns());
        }
        if (select->input()->kind()==PlanNode::Kind::WithColumn) {
            const auto* with_col=dynamic_cast<const WithColumnNode*>(select->input().get());
            if (!with_col) {
                return node;
            }
            const bool output_needs_with_col =
                std::find(select->columns().begin(), select->columns().end(), with_col->name()) !=
                select->columns().end();
            if (!output_needs_with_col) {
                return std::make_shared<SelectNamesNode>(with_col->input(), select->columns());
            }
            std::unordered_set<std::string> needed;
            for (const auto& col_name : select->columns()) {
                if (col_name!=with_col->name()) {
                    needed.insert(col_name);
                }
            }
            CollectColumnsFromExpr(with_col->expr().node_ptr(), &needed);
            const auto needed_cols=SortedColumns(needed);
            std::shared_ptr<PlanNode> narrowed_input=with_col->input();
            if (!IsExactSelectNames(narrowed_input, needed_cols)) {
                narrowed_input=std::make_shared<SelectNamesNode>(narrowed_input, needed_cols);
            }
            auto rebuilt_with_col =
                std::make_shared<WithColumnNode>(narrowed_input, with_col->name(), with_col->expr());
            return std::make_shared<SelectNamesNode>(rebuilt_with_col, select->columns());
        }
    }
    if (node->kind()==PlanNode::Kind::SelectExprs) {
        const auto* select=dynamic_cast<const SelectExprsNode*>(node.get());
        if (!select||!select->input()) {
            return node;
        }
        if (select->input()->kind()==PlanNode::Kind::Filter) {
            const auto* filter=dynamic_cast<const FilterNode*>(select->input().get());
            if (!filter) {
                return node;
            }
            std::unordered_set<std::string> needed;
            for (const auto& expr : select->expressions()) {
                CollectColumnsFromExpr(expr.node_ptr(), &needed);
            }
            CollectColumnsFromExpr(filter->predicate().node_ptr(), &needed);
            const auto needed_cols=SortedColumns(needed);
            std::shared_ptr<PlanNode> filtered_input=filter->input();
            if (!needed_cols.empty()&&!IsExactSelectNames(filtered_input, needed_cols)) {
                filtered_input=std::make_shared<SelectNamesNode>(filtered_input, needed_cols);
            }
            auto pushed_filter=std::make_shared<FilterNode>(filtered_input, filter->predicate());
            return std::make_shared<SelectExprsNode>(pushed_filter, select->expressions());
        }
    }
    return node;
}

std::shared_ptr<PlanNode> ApplyLimitPushdown(const std::shared_ptr<PlanNode>& node) {
    if (!node||node->kind()!=PlanNode::Kind::Head) {
        return node;
    }
    const auto* head=dynamic_cast<const HeadNode*>(node.get());
    if (!head||!head->input()) {
        return node;
    }
    if (head->input()->kind()==PlanNode::Kind::Head) {
        const auto* inner=dynamic_cast<const HeadNode*>(head->input().get());
        if (!inner) {
            return node;
        }
        return std::make_shared<HeadNode>(inner->input(), std::min(head->n(), inner->n()));
    }
    if (head->input()->kind()==PlanNode::Kind::SelectNames) {
        const auto* select=dynamic_cast<const SelectNamesNode*>(head->input().get());
        if (!select) {
            return node;
        }
        auto pushed_head=std::make_shared<HeadNode>(select->input(), head->n());
        return std::make_shared<SelectNamesNode>(pushed_head, select->columns());
    }
    if (head->input()->kind()==PlanNode::Kind::SelectExprs) {
        const auto* select=dynamic_cast<const SelectExprsNode*>(head->input().get());
        if (!select) {
            return node;
        }
        auto pushed_head=std::make_shared<HeadNode>(select->input(), head->n());
        return std::make_shared<SelectExprsNode>(pushed_head, select->expressions());
    }
    if (head->input()->kind()==PlanNode::Kind::WithColumn) {
        const auto* with_col=dynamic_cast<const WithColumnNode*>(head->input().get());
        if (!with_col) {
            return node;
        }
        auto pushed_head=std::make_shared<HeadNode>(with_col->input(), head->n());
        return std::make_shared<WithColumnNode>(pushed_head, with_col->name(), with_col->expr());
    }
    return node;
}

std::shared_ptr<PlanNode> MakePlanWithOptimizedExprs(const std::shared_ptr<PlanNode>& node) {
    if (!node) {
        return node;
    }
    switch (node->kind()) {
        case PlanNode::Kind::ScanCsv:
        case PlanNode::Kind::ScanParquet:
            return node;  // no expressions to optimize in leaf scan nodes
        case PlanNode::Kind::SelectNames: {
            const auto* n=dynamic_cast<const SelectNamesNode*>(node.get());
            if (!n) return node;
            return std::make_shared<SelectNamesNode>(MakePlanWithOptimizedExprs(n->input()), n->columns());
        }
        case PlanNode::Kind::SelectExprs: {
            const auto* n=dynamic_cast<const SelectExprsNode*>(node.get());
            if (!n) return node;
            std::vector<Expr> optimized;
            optimized.reserve(n->expressions().size());
            for (const auto& expr : n->expressions()) {
                optimized.push_back(OptimizeExpr(expr));
            }
            return std::make_shared<SelectExprsNode>(MakePlanWithOptimizedExprs(n->input()), optimized);
        }
        case PlanNode::Kind::Filter: {
            const auto* n=dynamic_cast<const FilterNode*>(node.get());
            if (!n) return node;
            return std::make_shared<FilterNode>(MakePlanWithOptimizedExprs(n->input()), OptimizeExpr(n->predicate()));
        }
        case PlanNode::Kind::WithColumn: {
            const auto* n=dynamic_cast<const WithColumnNode*>(node.get());
            if (!n) return node;
            return std::make_shared<WithColumnNode>(MakePlanWithOptimizedExprs(n->input()), n->name(), OptimizeExpr(n->expr()));
        }
        case PlanNode::Kind::GroupByAggregate: {
            const auto* n=dynamic_cast<const GroupByAggregateNode*>(node.get());
            if (!n) return node;
            GroupByAggregateNode::AggMap agg_map;
            for (const auto& kv : n->agg_map()) {
                agg_map.emplace(kv.first, OptimizeExpr(kv.second));
            }
            return std::make_shared<GroupByAggregateNode>(
                MakePlanWithOptimizedExprs(n->input()), n->keys(), std::move(agg_map));
        }
        case PlanNode::Kind::Join: {
            const auto* n=dynamic_cast<const JoinNode*>(node.get());
            if (!n) return node;
            return std::make_shared<JoinNode>(
                MakePlanWithOptimizedExprs(n->input()),
                MakePlanWithOptimizedExprs(n->right_input()),
                n->on(),
                n->how());
        }
        case PlanNode::Kind::Sort: {
            const auto* n=dynamic_cast<const SortNode*>(node.get());
            if (!n) return node;
            return std::make_shared<SortNode>(MakePlanWithOptimizedExprs(n->input()), n->columns(), n->ascending());
        }
        case PlanNode::Kind::Head: {
            const auto* n=dynamic_cast<const HeadNode*>(node.get());
            if (!n) return node;
            return std::make_shared<HeadNode>(MakePlanWithOptimizedExprs(n->input()), n->n());
        }
    }
    return node;
}

std::shared_ptr<PlanNode> ApplyAllRules(const std::shared_ptr<PlanNode>& node) {
    auto current=MakePlanWithOptimizedExprs(node);
    current=ApplyPredicatePushdown(current);
    current=ApplyProjectionPushdown(current);
    current=ApplyLimitPushdown(current);
    return current;
}

std::string PlanFingerprintImpl(const std::shared_ptr<PlanNode>& node,
                                std::unordered_map<const PlanNode*, std::string>* memo) {
    if (!node) {
        return "null";
    }
    auto it=memo->find(node.get());
    if (it!=memo->end()) {
        return it->second;
    }
    std::ostringstream out;
    out << static_cast<int>(node->kind()) << "|" << node->ToString()
        << "|" << PlanFingerprintImpl(node->input(), memo)
        << "|" << PlanFingerprintImpl(node->right_input(), memo);
    const std::string fp=out.str();
    memo->emplace(node.get(), fp);
    return fp;
}

std::string PlanFingerprint(const std::shared_ptr<PlanNode>& node) {
    std::unordered_map<const PlanNode*, std::string> memo;
    return PlanFingerprintImpl(node, &memo);
}

std::shared_ptr<PlanNode> OptimizeTreeOnce(const std::shared_ptr<PlanNode>& node) {
    if (!node) {
        return node;
    }
    std::shared_ptr<PlanNode> rebuilt;
    switch (node->kind()) {
        case PlanNode::Kind::ScanCsv:
        case PlanNode::Kind::ScanParquet:
            rebuilt=node;
            break;
        case PlanNode::Kind::SelectNames: {
            const auto* n=dynamic_cast<const SelectNamesNode*>(node.get());
            rebuilt=std::make_shared<SelectNamesNode>(OptimizeTreeOnce(n->input()), n->columns());
            break;
        }
        case PlanNode::Kind::SelectExprs: {
            const auto* n=dynamic_cast<const SelectExprsNode*>(node.get());
            rebuilt=std::make_shared<SelectExprsNode>(OptimizeTreeOnce(n->input()), n->expressions());
            break;
        }
        case PlanNode::Kind::Filter: {
            const auto* n=dynamic_cast<const FilterNode*>(node.get());
            rebuilt=std::make_shared<FilterNode>(OptimizeTreeOnce(n->input()), n->predicate());
            break;
        }
        case PlanNode::Kind::WithColumn: {
            const auto* n=dynamic_cast<const WithColumnNode*>(node.get());
            rebuilt=std::make_shared<WithColumnNode>(OptimizeTreeOnce(n->input()), n->name(), n->expr());
            break;
        }
        case PlanNode::Kind::GroupByAggregate: {
            const auto* n=dynamic_cast<const GroupByAggregateNode*>(node.get());
            rebuilt=std::make_shared<GroupByAggregateNode>(OptimizeTreeOnce(n->input()), n->keys(), n->agg_map());
            break;
        }
        case PlanNode::Kind::Join: {
            const auto* n=dynamic_cast<const JoinNode*>(node.get());
            rebuilt=std::make_shared<JoinNode>(
                OptimizeTreeOnce(n->input()), OptimizeTreeOnce(n->right_input()), n->on(), n->how());
            break;
        }
        case PlanNode::Kind::Sort: {
            const auto* n=dynamic_cast<const SortNode*>(node.get());
            rebuilt=std::make_shared<SortNode>(OptimizeTreeOnce(n->input()), n->columns(), n->ascending());
            break;
        }
        case PlanNode::Kind::Head: {
            const auto* n=dynamic_cast<const HeadNode*>(node.get());
            rebuilt=std::make_shared<HeadNode>(OptimizeTreeOnce(n->input()), n->n());
            break;
        }
    }
    return ApplyAllRules(rebuilt);
}

}  

std::shared_ptr<PlanNode> QueryOptimizer::optimize(const std::shared_ptr<PlanNode>& root) const {
    std::shared_ptr<PlanNode> current=root;
    std::string previous_fingerprint;
    constexpr int kMaxPasses=8;
    for (int pass=0; pass<kMaxPasses; ++pass) {
        current=OptimizeTreeOnce(current);
        const std::string fp=PlanFingerprint(current);
        if (fp==previous_fingerprint) {
            break;
        }
        previous_fingerprint=fp;
    }
    return current;
}

} 
