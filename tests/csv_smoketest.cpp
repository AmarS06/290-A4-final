#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include "dataframelib/dataframelib.h"

using namespace dataframelib;

#ifndef SMOKETEST_DATA_DIR
#define SMOKETEST_DATA_DIR "../data"
#endif
#ifndef SMOKETEST_GEN_DIR
#define SMOKETEST_GEN_DIR "../generated"
#endif

static const std::string kData(SMOKETEST_DATA_DIR);
static const std::string kGen(SMOKETEST_GEN_DIR);

static const std::vector<std::string> kOut = {
    "id","name","dept_name","age","salary","bonus","total_comp","location"
};

int main() {
    std::filesystem::create_directories(kData);
    std::filesystem::create_directories(kGen);

    const std::string emp_csv  = kData + "/employees.csv";
    const std::string dept_csv = kData + "/departments.csv";

    {
        std::ofstream f(emp_csv);
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
        std::ofstream f(dept_csv);
        f << "dept_id,dept_name,location\n"
          << "1,Engineering,Building A\n"
          << "2,HR,Building B\n"
          << "3,Finance,Building C\n";
    }
    std::cout << "Input data: " << kData << "\n";

    auto eager_result = read_csv(emp_csv)
        .filter(col("age") >= 28)
        .with_column("bonus", col("salary") * 0.15)
        .join(read_csv(dept_csv), {"dept_id"}, "inner")
        .with_column("total_comp", col("salary") + col("bonus"))
        .sort({"salary"}, false)
        .select(kOut);

    assert(eager_result.num_rows()==8);
    assert(eager_result.num_columns()==8);
    assert(eager_result.write_csv(kData+"/eager_output.csv").ok());
    std::cout << "Eager output: " << kData << "/eager_output.csv\n";

    auto lazy_plan = scan_csv(emp_csv)
        .sort({"salary"}, false)
        .filter(col("age") >= 28)
        .with_column("bonus", col("salary") * 0.15)
        .join(scan_csv(dept_csv), {"dept_id"}, "inner")
        .with_column("total_comp", col("salary") + col("bonus"))
        .select(kOut);

    assert(lazy_plan.explain(kGen+"/plan.png").ok());
    std::cout << "Logical DAG:   " << kGen << "/plan_logical.png\n";
    std::cout << "Optimized DAG: " << kGen << "/plan_optimized.png\n";

    auto lazy_result = lazy_plan.collect();
    assert(lazy_result.num_rows()==8);
    assert(lazy_result.num_columns()==8);
    assert(lazy_result.write_csv(kData+"/lazy_output.csv").ok());
    std::cout << "Lazy output:  " << kData << "/lazy_output.csv\n";

    // Constant filter elimination + dead WithColumn elimination
    auto dead_plan = scan_csv(emp_csv)
        .filter(lit(1) == lit(1))
        .with_column("double_salary", col("salary") * lit(2))
        .select(std::vector<Expr>{col("id").alias("eid"), col("name")});

    assert(dead_plan.explain(kGen+"/dead_plan.png").ok());
    std::cout << "Dead-col logical:   " << kGen << "/dead_plan_logical.png\n";
    std::cout << "Dead-col optimized: " << kGen << "/dead_plan_optimized.png\n";

    auto dead_result = dead_plan.collect();
    assert(dead_result.num_rows()==10);
    assert(dead_result.num_columns()==2);

    // GroupBy predicate pushdown check
    auto groupby_plan = scan_csv(emp_csv)
        .join(scan_csv(dept_csv), {"dept_id"}, "inner")
        .group_by({"dept_name"})
        .aggregate({{"salary", "mean"}})
        .filter(col("dept_name") == lit("Engineering"));

    assert(groupby_plan.explain(kGen+"/groupby_plan.png").ok());
    std::cout << "GroupBy logical:   " << kGen << "/groupby_plan_logical.png\n";
    std::cout << "GroupBy optimized: " << kGen << "/groupby_plan_optimized.png\n";

    auto groupby_result = groupby_plan.collect();
    assert(groupby_result.num_rows()==1);
    assert(groupby_result.num_columns()==2);

    assert(std::filesystem::exists(kData+"/eager_output.csv"));
    assert(std::filesystem::exists(kData+"/lazy_output.csv"));
    assert(std::filesystem::exists(kGen+"/plan_logical.png"));
    assert(std::filesystem::exists(kGen+"/plan_optimized.png"));
    assert(std::filesystem::exists(kGen+"/groupby_plan_logical.png"));
    assert(std::filesystem::exists(kGen+"/groupby_plan_optimized.png"));
    assert(std::filesystem::exists(kGen+"/dead_plan_logical.png"));
    assert(std::filesystem::exists(kGen+"/dead_plan_optimized.png"));

    std::cout << "All smoke tests passed.\n";
    return 0;
}
