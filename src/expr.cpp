#include "dataframelib/expr.hpp"
#include <cmath>
#include <stdexcept>
#include <arrow/scalar.h>

namespace dataframelib {

Expr::Expr(int value) : node_(lit(static_cast<int32_t>(value)).node_) {}
Expr::Expr(double value) : node_(lit(value).node_) {}
Expr::Expr(const char* value) : node_(lit(std::string(value)).node_) {}
Expr::Expr(std::string value) : node_(lit(std::move(value)).node_) {}

Expr Expr::operator+(const Expr& rhs) const {
    return Expr(std::make_shared<BinaryNode>(BinaryOpKind::kAdd, node_, rhs.node_));
}
Expr Expr::operator-(const Expr& rhs) const {
    return Expr(std::make_shared<BinaryNode>(BinaryOpKind::kSub, node_, rhs.node_));
}
Expr Expr::operator*(const Expr& rhs) const {
    return Expr(std::make_shared<BinaryNode>(BinaryOpKind::kMul, node_, rhs.node_));
}
Expr Expr::operator/(const Expr& rhs) const {
    return Expr(std::make_shared<BinaryNode>(BinaryOpKind::kDiv, node_, rhs.node_));
}
Expr Expr::operator%(const Expr& rhs) const {
    return Expr(std::make_shared<BinaryNode>(BinaryOpKind::kMod, node_, rhs.node_));
}

Expr Expr::operator==(const Expr& rhs) const {
    return Expr(std::make_shared<BinaryNode>(BinaryOpKind::kEq, node_, rhs.node_));
}
Expr Expr::operator!=(const Expr& rhs) const {
    return Expr(std::make_shared<BinaryNode>(BinaryOpKind::kNeq, node_, rhs.node_));
}
Expr Expr::operator<(const Expr& rhs) const {
    return Expr(std::make_shared<BinaryNode>(BinaryOpKind::kLt, node_, rhs.node_));
}
Expr Expr::operator<=(const Expr& rhs) const {
    return Expr(std::make_shared<BinaryNode>(BinaryOpKind::kLe, node_, rhs.node_));
}
Expr Expr::operator>(const Expr& rhs) const {
    return Expr(std::make_shared<BinaryNode>(BinaryOpKind::kGt, node_, rhs.node_));
}
Expr Expr::operator>=(const Expr& rhs) const {
    return Expr(std::make_shared<BinaryNode>(BinaryOpKind::kGe, node_, rhs.node_));
}

Expr Expr::operator&(const Expr& rhs) const {
    return Expr(std::make_shared<BinaryNode>(BinaryOpKind::kAnd, node_, rhs.node_));
}
Expr Expr::operator|(const Expr& rhs) const {
    return Expr(std::make_shared<BinaryNode>(BinaryOpKind::kOr, node_, rhs.node_));
}
Expr Expr::operator~() const {
    return Expr(std::make_shared<UnaryNode>(UnaryOpKind::kNot, node_));
}

Expr Expr::is_null() const {
    return Expr(std::make_shared<UnaryNode>(UnaryOpKind::kIsNull, node_));
}
Expr Expr::is_not_null() const {
    return Expr(std::make_shared<UnaryNode>(UnaryOpKind::kIsNotNull, node_));
}
Expr Expr::abs() const {
    return Expr(std::make_shared<UnaryNode>(UnaryOpKind::kAbs, node_));
}

Expr Expr::length() const {
    return Expr(std::make_shared<StringOpNode>(StringOpKind::kLength, node_));
}
Expr Expr::contains(std::string s) const {
    return Expr(std::make_shared<StringOpNode>(StringOpKind::kContains, node_, std::move(s)));
}
Expr Expr::starts_with(std::string s) const {
    return Expr(std::make_shared<StringOpNode>(StringOpKind::kStartsWith, node_, std::move(s)));
}
Expr Expr::ends_with(std::string s) const {
    return Expr(std::make_shared<StringOpNode>(StringOpKind::kEndsWith, node_, std::move(s)));
}
Expr Expr::to_lower() const {
    return Expr(std::make_shared<StringOpNode>(StringOpKind::kToLower, node_));
}
Expr Expr::to_upper() const {
    return Expr(std::make_shared<StringOpNode>(StringOpKind::kToUpper, node_));
}

Expr Expr::sum() const   { return Expr(std::make_shared<AggNode>(AggOpKind::kSum,   node_)); }
Expr Expr::mean() const  { return Expr(std::make_shared<AggNode>(AggOpKind::kMean,  node_)); }
Expr Expr::count() const { return Expr(std::make_shared<AggNode>(AggOpKind::kCount, node_)); }
Expr Expr::min() const   { return Expr(std::make_shared<AggNode>(AggOpKind::kMin,   node_)); }
Expr Expr::max() const   { return Expr(std::make_shared<AggNode>(AggOpKind::kMax,   node_)); }

Expr Expr::alias(std::string name) const {
    return Expr(std::make_shared<AliasNode>(std::move(name), node_));
}

std::string Expr::ToString() const {
    return node_->ToString();
}

Expr col(std::string name) {
    return Expr(std::make_shared<ColNode>(std::move(name)));
}

Expr lit(int32_t value) {
    return Expr(std::make_shared<LitNode>(std::make_shared<arrow::Int32Scalar>(value)));
}
Expr lit(int64_t value) {
    return Expr(std::make_shared<LitNode>(std::make_shared<arrow::Int64Scalar>(value)));
}
Expr lit(float value) {
    if (std::isnan(value)) {
        throw std::invalid_argument("lit: NaN is not allowed; represent missing values using null.");
    }
    return Expr(std::make_shared<LitNode>(std::make_shared<arrow::FloatScalar>(value)));
}
Expr lit(double value) {
    if (std::isnan(value)) {
        throw std::invalid_argument("lit: NaN is not allowed; represent missing values using null.");
    }
    return Expr(std::make_shared<LitNode>(std::make_shared<arrow::DoubleScalar>(value)));
}
Expr lit(bool value) {
    return Expr(std::make_shared<LitNode>(std::make_shared<arrow::BooleanScalar>(value)));
}
Expr lit(std::string value) {
    return Expr(std::make_shared<LitNode>(std::make_shared<arrow::StringScalar>(std::move(value))));
}
Expr lit(const char* value) {
    return lit(std::string(value));
}

std::string LitNode::ToString() const {
    if (!scalar_ || !scalar_->is_valid) {
        return "lit(null)";
    }
    if (scalar_->type->id() == arrow::Type::STRING) {
        return "lit(\"" + scalar_->ToString() + "\")";
    }
    return "lit(" + scalar_->ToString() + ")";
}

std::string BinaryNode::ToString() const {
    static const char* const kSymbols[] = {
        "+", "-", "*", "/", "%",           
        "==", "!=", "<", "<=", ">", ">=",  
        "&", "|",                           
    };
    const int idx = static_cast<int>(op_);
    return "(" + left_->ToString() + " " + kSymbols[idx] + " " +
           right_->ToString() + ")";
}

std::string UnaryNode::ToString() const {
    switch (op_) {
        case UnaryOpKind::kNot:
            return "!(" + input_->ToString() + ")";
        case UnaryOpKind::kAbs:
            return "abs(" + input_->ToString() + ")";
        case UnaryOpKind::kIsNull:
            return input_->ToString() + ".is_null()";
        case UnaryOpKind::kIsNotNull:
            return input_->ToString() + ".is_not_null()";
    }
    return "";
}

std::string StringOpNode::ToString() const {
    switch (op_) {
        case StringOpKind::kLength:
            return input_->ToString() + ".length()";
        case StringOpKind::kContains:
            return input_->ToString() + ".contains(\"" + arg_ + "\")";
        case StringOpKind::kStartsWith:
            return input_->ToString() + ".starts_with(\"" + arg_ + "\")";
        case StringOpKind::kEndsWith:
            return input_->ToString() + ".ends_with(\"" + arg_ + "\")";
        case StringOpKind::kToLower:
            return input_->ToString() + ".to_lower()";
        case StringOpKind::kToUpper:
            return input_->ToString() + ".to_upper()";
    }
    return "";
}

std::string AggNode::ToString() const {
    switch (op_) {
        case AggOpKind::kSum:   return input_->ToString() + ".sum()";
        case AggOpKind::kMean:  return input_->ToString() + ".mean()";
        case AggOpKind::kCount: return input_->ToString() + ".count()";
        case AggOpKind::kMin:   return input_->ToString() + ".min()";
        case AggOpKind::kMax:   return input_->ToString() + ".max()";
    }
    return "";
}

}  