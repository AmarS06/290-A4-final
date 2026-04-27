#pragma once

#include <memory>
#include <string>
#include <arrow/scalar.h>

namespace dataframelib {

class ExprNode;

enum class BinaryOpKind {
    kAdd, kSub, kMul, kDiv, kMod, kEq, kNeq, kLt, kLe, kGt, kGe, kAnd, kOr,
};

enum class UnaryOpKind {
    kNot, kAbs, kIsNull, kIsNotNull,
};

enum class StringOpKind {
    kLength, kContains, kStartsWith, kEndsWith, kToLower, kToUpper,
};

enum class AggOpKind {
    kSum, kMean, kCount, kMin, kMax,
};

class Expr {
public:
    Expr(int value);
    Expr(double value);
    Expr(const char* value);
    Expr(std::string value);

    // Access the underlying node
    const ExprNode& node() const { return *node_; }
    std::shared_ptr<ExprNode> node_ptr() const { return node_; }

    // Arithmetic
    Expr operator+(const Expr& rhs) const;
    Expr operator-(const Expr& rhs) const;
    Expr operator*(const Expr& rhs) const;
    Expr operator/(const Expr& rhs) const;
    Expr operator%(const Expr& rhs) const;

    // Comparison
    Expr operator==(const Expr& rhs) const;
    Expr operator!=(const Expr& rhs) const;
    Expr operator<(const Expr& rhs) const;
    Expr operator<=(const Expr& rhs) const;
    Expr operator>(const Expr& rhs) const;
    Expr operator>=(const Expr& rhs) const;

    // Boolean logical
    Expr operator&(const Expr& rhs) const;
    Expr operator|(const Expr& rhs) const;
    Expr operator~() const;

    // Null predicates
    Expr is_null() const;
    Expr is_not_null() const;

    // Unary numeric
    Expr abs() const;

    // String functions
    Expr length() const;
    Expr contains(std::string s) const;
    Expr starts_with(std::string s) const;
    Expr ends_with(std::string s) const;
    Expr to_lower() const;
    Expr to_upper() const;

    // Aggregation expressions
    Expr sum() const;
    Expr mean() const;
    Expr count() const;
    Expr min() const;
    Expr max() const;

    // Rename the output
    Expr alias(std::string name) const;

    // Human-readable representation
    std::string ToString() const;

private:
    explicit Expr(std::shared_ptr<ExprNode> node) : node_(std::move(node)) {}

    std::shared_ptr<ExprNode> node_;

    friend Expr col(std::string name);
    friend Expr lit(int32_t value);
    friend Expr lit(int64_t value);
    friend Expr lit(float value);
    friend Expr lit(double value);
    friend Expr lit(bool value);
    friend Expr lit(std::string value);
    friend Expr lit(const char* value);
};

Expr col(std::string name);
Expr lit(int32_t value);
Expr lit(int64_t value);
Expr lit(float value);
Expr lit(double value);
Expr lit(bool value);
Expr lit(std::string value);
Expr lit(const char* value);

class ExprNode {
public:
    enum class Kind { Col, Lit, Alias, Binary, Unary, StringOp, Agg };
    virtual Kind kind() const = 0;
    virtual std::string ToString() const = 0;
    virtual ~ExprNode() = default;
};

class ColNode : public ExprNode {
public:
    explicit ColNode(std::string name) : name_(std::move(name)) {}
    Kind kind() const override { return Kind::Col; }
    std::string ToString() const override { return "col(\"" + name_ + "\")"; }
    const std::string& name() const { return name_; }
private:
    std::string name_;
};

class LitNode : public ExprNode {
public:
    explicit LitNode(std::shared_ptr<arrow::Scalar> scalar)
        : scalar_(std::move(scalar)) {}
    Kind kind() const override { return Kind::Lit; }
    std::string ToString() const override;
    const std::shared_ptr<arrow::Scalar>& scalar() const { return scalar_; }
private:
    std::shared_ptr<arrow::Scalar> scalar_;
};

class AliasNode : public ExprNode {
public:
    AliasNode(std::string name, std::shared_ptr<ExprNode> input)
        : name_(std::move(name)), input_(std::move(input)) {}
    Kind kind() const override { return Kind::Alias; }
    std::string ToString() const override { return input_->ToString() + ".alias(\"" + name_ + "\")"; }
    const std::string& name() const { return name_; }
    const std::shared_ptr<ExprNode>& input() const { return input_; }
private:
    std::string name_;
    std::shared_ptr<ExprNode> input_;
};

class BinaryNode : public ExprNode {
public:
    BinaryNode(BinaryOpKind op, std::shared_ptr<ExprNode> left, std::shared_ptr<ExprNode> right)
        : op_(op), left_(std::move(left)), right_(std::move(right)) {}
    Kind kind() const override { return Kind::Binary; }
    std::string ToString() const override;
    BinaryOpKind op() const { return op_; }
    const std::shared_ptr<ExprNode>& left() const { return left_; }
    const std::shared_ptr<ExprNode>& right() const { return right_; }
private:
    BinaryOpKind op_;
    std::shared_ptr<ExprNode> left_;
    std::shared_ptr<ExprNode> right_;
};

class UnaryNode : public ExprNode {
public:
    UnaryNode(UnaryOpKind op, std::shared_ptr<ExprNode> input)
        : op_(op), input_(std::move(input)) {}
    Kind kind() const override { return Kind::Unary; }
    std::string ToString() const override;
    UnaryOpKind op() const { return op_; }
    const std::shared_ptr<ExprNode>& input() const { return input_; }
private:
    UnaryOpKind op_;
    std::shared_ptr<ExprNode> input_;
};

class StringOpNode : public ExprNode {
public:
    StringOpNode(StringOpKind op, std::shared_ptr<ExprNode> input, std::string arg = "")
        : op_(op), input_(std::move(input)), arg_(std::move(arg)) {}
    Kind kind() const override { return Kind::StringOp; }
    std::string ToString() const override;
    StringOpKind op() const { return op_; }
    const std::shared_ptr<ExprNode>& input() const { return input_; }
    const std::string& arg() const { return arg_; }
private:
    StringOpKind op_;
    std::shared_ptr<ExprNode> input_;
    std::string arg_;
};

class AggNode : public ExprNode {
public:
    AggNode(AggOpKind op, std::shared_ptr<ExprNode> input)
        : op_(op), input_(std::move(input)) {}
    Kind kind() const override { return Kind::Agg; }
    std::string ToString() const override;
    AggOpKind op() const { return op_; }
    const std::shared_ptr<ExprNode>& input() const { return input_; }
private:
    AggOpKind op_;
    std::shared_ptr<ExprNode> input_;
};

} 