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
  "${HOMEBREW_PREFIX}/opt/hiredis"
  "${HOMEBREW_PREFIX}/opt/grpc"
  "${HOMEBREW_PREFIX}/opt/protobuf"
  "${HOMEBREW_PREFIX}/opt/abseil"
)

# ── Third-party packages ──────────────────────────────────────────────────────
find_package(nlohmann_json CONFIG REQUIRED)
find_package(prometheus-cpp CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)

# hiredis ships only pkg-config, not a CMake config file.
find_library(HIREDIS_LIB hiredis HINTS "${HOMEBREW_PREFIX}/opt/hiredis/lib" REQUIRED)
find_path(HIREDIS_INCLUDE hiredis/hiredis.h HINTS "${HOMEBREW_PREFIX}/opt/hiredis/include" REQUIRED)
add_library(hiredis::hiredis UNKNOWN IMPORTED)
set_target_properties(hiredis::hiredis PROPERTIES
    IMPORTED_LOCATION             "${HIREDIS_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${HIREDIS_INCLUDE}"
)

# ── llama.cpp ─────────────────────────────────────────────────────────────────
include(FetchContent)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)
set(GGML_METAL           ON  CACHE BOOL "" FORCE)
set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
    GIT_TAG        b5159
    GIT_SHALLOW    TRUE
    GIT_SUBMODULES ""
)
FetchContent_MakeAvailable(llama_cpp)

# ── Protobuf / gRPC code generation ──────────────────────────────────────────
set(PROTO_DIR    "${CMAKE_CURRENT_SOURCE_DIR}/proto")
set(PROTO_FILE   "${PROTO_DIR}/inference.proto")
set(GENERATED    "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${GENERATED}")

add_custom_command(
    OUTPUT
        "${GENERATED}/inference.pb.cc"
        "${GENERATED}/inference.pb.h"
        "${GENERATED}/inference.grpc.pb.cc"
        "${GENERATED}/inference.grpc.pb.h"
    COMMAND $<TARGET_FILE:protobuf::protoc>
    ARGS
        --cpp_out  "${GENERATED}"
        --grpc_out "${GENERATED}"
        --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
        -I "${PROTO_DIR}"
        "${PROTO_FILE}"
    DEPENDS "${PROTO_FILE}"
    COMMENT "Generating protobuf + gRPC stubs from inference.proto"
)

add_library(inference_proto
    "${GENERATED}/inference.pb.cc"
    "${GENERATED}/inference.grpc.pb.cc"
)
target_include_directories(inference_proto PUBLIC "${GENERATED}")
target_link_libraries(inference_proto
    PUBLIC
        protobuf::libprotobuf
        gRPC::grpc++
)

# ── Core library ──────────────────────────────────────────────────────────────
add_library(inference_core
    src/request.cpp
    src/metrics.cpp
    src/rate_limiter.cpp
    src/kv_slot_manager.cpp
    src/continuous_batch_engine.cpp
)
target_include_directories(inference_core PUBLIC include)
target_link_libraries(inference_core
    PUBLIC
        nlohmann_json::nlohmann_json
        prometheus-cpp::pull
        spdlog::spdlog
        llama
        hiredis::hiredis
)

# ── gRPC server library ───────────────────────────────────────────────────────
add_library(grpc_server_lib src/grpc_server.cpp)
target_include_directories(grpc_server_lib PUBLIC include)
target_link_libraries(grpc_server_lib
    PUBLIC
        inference_core
        inference_proto
        gRPC::grpc++
)

# ── Server executable ─────────────────────────────────────────────────────────
add_executable(inference_server src/main.cpp src/server.cpp)
target_link_libraries(inference_server
    PRIVATE
        inference_core
        grpc_server_lib
)

# ── Tests ─────────────────────────────────────────────────────────────────────
enable_testing()
add_executable(tests
    tests/test_rate_limiter.cpp
    tests/test_server.cpp
    tests/test_kv_slot_manager.cpp
    src/server.cpp
)
target_link_libraries(tests
    PRIVATE
        inference_core
        GTest::gtest_main
)
include(GoogleTest)
gtest_discover_tests(tests)

# ── Compiler warnings ─────────────────────────────────────────────────────────
target_compile_options(inference_core    PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(inference_server  PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(grpc_server_lib  PRIVATE -Wall -Wextra -Wpedantic)
