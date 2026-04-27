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


USING THE LIBRARY IN YOUR PROJECT
----------------------------------
After installation, add to your CMakeLists.txt:

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


CLEANUP
-------
Remove all build artefacts:

  rm -rf build
