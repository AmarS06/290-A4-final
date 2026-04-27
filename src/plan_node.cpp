#include "dataframelib/plan_node.hpp"
#include <unordered_map>
#include <vector>
#include <sstream>

namespace dataframelib {

namespace {
std::string EscapeDotLabel(const std::string& raw) {
    std::string escaped;
    escaped.reserve(raw.size());
    for (char c : raw) {
        if (c == '\\' || c == '"') {
            escaped.push_back('\\');
            escaped.push_back(c);
        } else if (c == '\n') {
            escaped += "\\n";
        } else {
            escaped.push_back(c);
        }
    }
    return escaped;
}
}

void BuildDotGraph(const std::shared_ptr<PlanNode>& root, std::string* dot_graph_out) {
    std::unordered_map<const PlanNode*, std::size_t> node_ids;
    std::vector<std::shared_ptr<PlanNode>> stack = {root};
    std::vector<std::shared_ptr<PlanNode>> ordered_nodes;

    while (!stack.empty()) {
        std::shared_ptr<PlanNode> current = stack.back();
        stack.pop_back();
        if (!current) continue;
        const PlanNode* key = current.get();
        if (node_ids.find(key) != node_ids.end()) continue;
        node_ids.emplace(key, ordered_nodes.size());
        ordered_nodes.push_back(current);
        if (current->input()) stack.push_back(current->input());
        if (current->right_input()) stack.push_back(current->right_input());
    }

    std::ostringstream dot;
    dot << "digraph LazyPlan {\n";
    dot << "  rankdir=TB;\n";
    dot << "  node [shape=box, style=rounded, fontsize=10];\n";

    for (const auto& node : ordered_nodes) {
        const std::size_t id = node_ids[node.get()];
        dot << "  n" << id << " [label=\""
            << EscapeDotLabel(node->ToString()) << "\"];\n";
    }

    for (const auto& node : ordered_nodes) {
        const std::size_t child_id = node_ids[node.get()];
        if (node->input()) {
            const auto found = node_ids.find(node->input().get());
            if (found != node_ids.end()) {
                dot << "  n" << found->second << " -> n" << child_id << ";\n";
            }
        }
        if (node->right_input()) {
            const auto found = node_ids.find(node->right_input().get());
            if (found != node_ids.end()) {
                dot << "  n" << found->second << " -> n" << child_id << ";\n";
            }
        }
    }

    dot << "}\n";
    *dot_graph_out = dot.str();
}

}

