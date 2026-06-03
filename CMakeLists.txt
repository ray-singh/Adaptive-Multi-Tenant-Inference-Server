cmake_minimum_required(VERSION 3.20)
project(inference_server VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Homebrew prefix (Apple Silicon)
set(HOMEBREW_PREFIX "/opt/homebrew" CACHE PATH "Homebrew install prefix")
list(APPEND CMAKE_PREFIX_PATH
  "${HOMEBREW_PREFIX}/opt/nlohmann-json"
  "${HOMEBREW_PREFIX}/opt/spdlog"
  "${HOMEBREW_PREFIX}/opt/googletest"
  "${HOMEBREW_PREFIX}/opt/prometheus-cpp"
)

# Dependencies (cpp-httplib is vendored as include/httplib.h)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(prometheus-cpp CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)

# Core library (scheduler, workers, queue — shared by server and tests)
add_library(inference_core
  src/request.cpp
  src/request_queue.cpp
  src/scheduler.cpp
  src/worker_pool.cpp
  src/metrics.cpp
)
target_include_directories(inference_core PUBLIC include)
target_link_libraries(inference_core
  PUBLIC
    nlohmann_json::nlohmann_json
    prometheus-cpp::pull
    spdlog::spdlog
)

# Server executable
add_executable(inference_server src/main.cpp src/server.cpp)
target_link_libraries(inference_server
  PRIVATE
    inference_core
)

# Tests
enable_testing()
add_executable(tests
  tests/test_request_queue.cpp
  tests/test_scheduler.cpp
  tests/test_worker_pool.cpp
)
target_link_libraries(tests
  PRIVATE
    inference_core
    GTest::gtest_main
)
include(GoogleTest)
gtest_discover_tests(tests)

# Compiler warnings
target_compile_options(inference_core PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(inference_server PRIVATE -Wall -Wextra -Wpedantic)
