DataFrameLib
============
A high-performance C++20 DataFrame library with eager and lazy execution
modes, backed by Apache Arrow.

DEPENDENCIES
------------
The following must be installed before building:

  - CMake >= 3.20
  - A C++20 compiler: GCC >= 11  or  Clang >= 13
  - Apache Arrow >= 12 with Parquet support
  - Graphviz (the "dot" command) -- required at runtime for explain()

Installing Arrow (choose one):

  Homebrew (macOS):
    brew install apache-arrow

  Conda:
    conda install -c conda-forge arrow-cpp parquet-cpp

  APT (Ubuntu/Debian):
    sudo apt install libarrow-dev libparquet-dev

  From source:
    https://arrow.apache.org/docs/cpp/build_system.html

Installing Graphviz:

  Homebrew:  brew install graphviz
  APT:       sudo apt install graphviz


BUILD
-----
From the root of the project:

  cmake -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j$(nproc)      # Linux
  cmake --build build -j$(sysctl -n hw.logicalcpu)   # macOS

This produces:
  build/libdataframelib.a      -- static library


INSTALLATION
------------
  cmake --install build --prefix /usr/local

This copies:
  /usr/local/lib/libdataframelib.a
  /usr/local/include/dataframelib/   (all public headers)

To install to a custom prefix, replace /usr/local with any path, e.g.:
  cmake --install build --prefix $HOME/.local


USING THE LIBRARY
-----------------
After installation, add to CMakeLists.txt:

  find_package(Arrow REQUIRED)
  find_package(Parquet REQUIRED)

  target_include_directories(your_target PRIVATE /usr/local/include)
  target_link_libraries(your_target PRIVATE
      /usr/local/lib/libdataframelib.a
      Arrow::arrow_shared
      Parquet::parquet_shared)

Then include the umbrella header:

  #include "dataframelib/dataframelib.h"
  using namespace dataframelib;


QUICK EXAMPLE
-------------
Eager mode:

  auto df = read_csv("data.csv");
  auto result = df.filter(col("age") > 30)
                  .select({"name", "salary"});
  result.write_csv("output.csv");

Lazy mode (with query optimiser):

  auto df = scan_parquet("data.parquet");
  auto result = df
      .filter(col("age") > 30)
      .group_by({"dept"})
      .aggregate({{"salary", "mean"}});

  result.explain("plan.png");   // writes plan_logical.png + plan_optimized.png
  auto output = result.collect();
  output.write_parquet("summary.parquet");


GENERATING SAMPLE DATA AND QUERY-PLAN PNGs
------------------------------------------
To exercise the library and produce output data files and optimised-plan
PNGs, write a driver program (e.g. smoketest.cpp) and wire it into the
build:

  1. Create smoketest.cpp in the project root (or any subdirectory).
     Example contents:

       #include "dataframelib/dataframelib.h"
       #include <filesystem>
       using namespace dataframelib;
       int main() {
           std::filesystem::create_directories("data");
           std::filesystem::create_directories("generated");
           // --- write sample CSV ---
           // (see QUICK EXAMPLE above for API usage)
           auto df = scan_csv("data/input.csv")
               .filter(col("age") > 30)
               .select({"name", "salary"});
           df.explain("generated/plan.png");  // plan_logical + plan_optimized
           df.collect().write_csv("data/output.csv");
       }

  NOTE -- assert() is disabled in Release builds:
  CMake's Release mode adds -DNDEBUG, which silently turns every assert()
  into a no-op.  A smoketest that uses assert() will appear to pass even
  when writes fail or row counts are wrong.  Use one of these instead:

    Option A -- explicit checks (recommended, works in any build type):

       static void check(bool ok, const char* msg) {
           if (!ok) throw std::runtime_error(msg);
       }
       static void check_status(const arrow::Status& s, const char* msg) {
           if (!s.ok())
               throw std::runtime_error(std::string(msg) + ": " + s.ToString());
       }
       // then in main():
       check_status(df.write_csv("data/out.csv"), "write_csv");
       check(std::filesystem::exists("data/out.csv"), "file not created");

    Option B -- undefine NDEBUG for the smoketest target only:

       target_compile_options(smoketest PRIVATE -UNDEBUG)

    The smoketest.cpp included in this project uses Option A together
    with Option B.

  2. Add the following lines to CMakelists.txt (after the install() block):

       add_executable(smoketest smoketest.cpp)
       target_link_libraries(smoketest PRIVATE dataframelib)
       target_compile_options(smoketest PRIVATE -UNDEBUG)

  3. Full pipeline -- build, run, inspect, then clean up:

       # build
       cmake -B build -DCMAKE_BUILD_TYPE=Release
       cmake --build build -j$(sysctl -n hw.logicalcpu)   # macOS
       cmake --build build -j$(nproc)                      # Linux

       # run (writes to data/ and generated/)
       ./build/smoketest

       # inspect outputs
       ls data/ generated/

       # remove all generated artefacts when done
       rm -rf build data generated

  Output files produced by ./build/smoketest:
    data/employees.csv            -- generated input data
    data/departments.csv          -- generated input data
    data/eager_output.csv         -- result of the eager query
    data/lazy_output.csv          -- result of the lazy query
    generated/plan_logical.png    -- unoptimised query-plan DAG
    generated/plan_optimized.png  -- optimised query-plan DAG
    generated/join_split_logical.png   -- join predicate split (before)
    generated/join_split_optimized.png -- join predicate split (after)

  Graphviz must be installed for the PNG files to be generated
  (see DEPENDENCIES above).

  Important: data/ and generated/ are created by the smoketest at
  runtime.  Remove them before submission or packaging:

       rm -rf data generated


CLEANUP
-------
Remove all build and runtime artefacts before packaging for submission:

  rm -rf build data generated
