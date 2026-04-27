#pragma once
#include <memory>
#include <string>
#include <arrow/type.h>
namespace dataframelib {
enum class DType {
    kInt32,
    kInt64,
    kFloat32,
    kFloat64,
    kString,
    kBoolean,
};

bool IsSupportedType(const std::shared_ptr<arrow::DataType>& type);
bool IsNumericType(const std::shared_ptr<arrow::DataType>& type);
bool IsFloatingType(const std::shared_ptr<arrow::DataType>& type);
DType ToDType(const std::shared_ptr<arrow::DataType>& type);
std::string ToString(DType dtype);
}  
