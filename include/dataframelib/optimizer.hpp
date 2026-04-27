#pragma once

#include <memory>

#include "dataframelib/plan_node.hpp"

namespace dataframelib {

class QueryOptimizer {
public:
    std::shared_ptr<PlanNode> optimize(const std::shared_ptr<PlanNode>& root) const;
};

}  
