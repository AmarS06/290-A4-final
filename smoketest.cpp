#include "dataframelib/dataframelib.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace dataframelib;

static void check(bool condition, const char* msg) {
    if (!condition) throw std::runtime_error(msg);
}

static void check_status(const arrow::Status& s, const char* msg) {
    if (!s.ok()) throw std::runtime_error(std::string(msg) + ": " + s.ToString());
}

int main() {
    try {
        std::filesystem::create_directories("data");
        std::filesystem::create_directories("generated");

        {
            std::ofstream f("data/employees.csv");
            f << "id,name,dept_id,age,salary\n"
              << "1,Alice,1,30,95000\n"
              << "2,Bob,1,25,72000\n"
              << "3,Carol,2,35,61000\n"
              << "4,Dave,1,28,85000\n"
              << "5,Eve,2,40,67000\n"
              << "6,Frank,3,33,88000\n"
              << "7,Grace,3,29,91000\n"
              << "8,Hank,2,26,58000\n"
              << "9,Ivy,1,31,99000\n"
              << "10,Jack,3,38,105000\n";
        }
        {
            std::ofstream f("data/departments.csv");
            f << "dept_id,dept_name,location\n"
              << "1,Engineering,Building A\n"
              << "2,HR,Building B\n"
              << "3,Finance,Building C\n";
        }

        auto eager = read_csv("data/employees.csv")
            .filter(col("age") >= 28)
            .with_column("bonus", col("salary") * 0.15)
            .join(read_csv("data/departments.csv"), {"dept_id"}, "inner")
            .with_column("total_comp", col("salary") + col("bonus"))
            .sort({"salary"}, false)
            .select({"id","name","dept_name","age","salary","bonus","total_comp","location"});

        check(eager.num_rows() == 8,    "eager: expected 8 rows");
        check(eager.num_columns() == 8, "eager: expected 8 columns");
        check_status(eager.write_csv("data/eager_output.csv"), "eager write_csv");
        check(std::filesystem::exists("data/eager_output.csv"), "eager_output.csv not created");
        std::cout << "Eager output:     data/eager_output.csv\n";

        auto lazy = scan_csv("data/employees.csv")
            .sort({"salary"}, false)
            .filter(col("age") >= 28)
            .with_column("bonus", col("salary") * 0.15)
            .join(scan_csv("data/departments.csv"), {"dept_id"}, "inner")
            .with_column("total_comp", col("salary") + col("bonus"))
            .select({"id","name","dept_name","age","salary","bonus","total_comp","location"});

        check_status(lazy.explain("generated/plan.png"), "lazy explain");
        check(std::filesystem::exists("generated/plan_logical.png"),  "plan_logical.png not created");
        check(std::filesystem::exists("generated/plan_optimized.png"), "plan_optimized.png not created");
        std::cout << "Logical plan:     generated/plan_logical.png\n";
        std::cout << "Optimized plan:   generated/plan_optimized.png\n";

        auto lazy_out = lazy.collect();
        check(lazy_out.num_rows() == 8,    "lazy: expected 8 rows");
        check(lazy_out.num_columns() == 8, "lazy: expected 8 columns");
        check_status(lazy_out.write_csv("data/lazy_output.csv"), "lazy write_csv");
        check(std::filesystem::exists("data/lazy_output.csv"), "lazy_output.csv not created");
        std::cout << "Lazy output:      data/lazy_output.csv\n";

        auto join_split = scan_csv("data/employees.csv")
            .select({"id","name","dept_id","age","salary"})
            .join(
                scan_csv("data/departments.csv").select({"dept_id","dept_name","location"}),
                {"dept_id"}, "inner")
            .filter(col("age") > lit(30) & col("dept_name").starts_with("E"));

        check_status(join_split.explain("generated/join_split.png"), "join_split explain");
        check(std::filesystem::exists("generated/join_split_logical.png"),  "join_split_logical.png not created");
        check(std::filesystem::exists("generated/join_split_optimized.png"), "join_split_optimized.png not created");
        std::cout << "Join-split plan:  generated/join_split_logical.png\n";
        std::cout << "Join-split opt:   generated/join_split_optimized.png\n";

        auto js_out = join_split.collect();
        check(js_out.num_rows() == 1,    "join_split: expected 1 row");
        check(js_out.num_columns() == 7, "join_split: expected 7 columns");

        std::cout << "\nAll checks passed.\n";

    } catch (const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
