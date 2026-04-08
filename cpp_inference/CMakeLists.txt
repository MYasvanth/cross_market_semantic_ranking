cmake_minimum_required(VERSION 3.20)
project(cross_market_inference)

set(CMAKE_CXX_STANDARD 17)

find_package(onnxruntime REQUIRED)
find_package(nlohmann_json REQUIRED)

add_executable(inference inference.cpp features.json)
target_link_libraries(inference onnxruntime nlohmann_json::nlohmann_json)
