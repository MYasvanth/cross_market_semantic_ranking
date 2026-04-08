#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <nlohmann/json.hpp>

// Strip newlines and control characters to prevent log injection
std::string sanitize_log(const std::string& input) {
    std::string out;
    out.reserve(input.size());
    std::copy_if(input.begin(), input.end(), std::back_inserter(out),
        [](unsigned char c) { return c >= 32 && c != 127; });
    return out;
}

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ranking_inference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    
    const std::string model_path = "artifacts/ranker.onnx";
    Ort::Session session(env, model_path.c_str(), session_options);
    
    // Sample features JSON
    std::ifstream f("cpp_inference/features.json");
    auto features = nlohmann::json::parse(f)["features"].get<std::vector<float>>();
    
    // Input tensor
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(features.size())};
    Ort::MemoryInfo memory_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, features.data(), features.size(), 
        input_shape.data(), input_shape.size()
    );
    
    const char* input_names[] = {"input_features"};
    const char* output_names[] = {"output_scores"};
    
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, 
        output_names, 1
    );
    
    float* scores = output_tensors[0].GetTensorMutableData<float>();
    std::cout << "Top score: " << sanitize_log(std::to_string(*scores)) << std::endl;
    
    return 0;
}

