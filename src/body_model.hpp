#pragma once
#include "body_pipeline.hpp"
#include <mutex>
namespace sam3d {
struct body_inference_options {
    uint32_t crop_size=512, intermediate_mask=31;
    bool correctives=true, slim_intermediates=false;
};
void validate_body_inference_options(body_inference_options);
// Checked GGUF-only pose-branch session. No reference/fixture directory, Python,
// TorchScript or inline parameters are runtime inputs. One-person no-mask mode.
class body_model {
public:
    body_model(const std::filesystem::path &backbone,const std::filesystem::path &branch,
               const std::filesystem::path &mhr,const std::string &module,const std::string &backend,
               uint32_t device,uint32_t threads,const std::string &description={},bool bf16=false,
               body_inference_options inference={});
    named_floats infer_rgb(std::span<const uint8_t> rgb,uint32_t width,uint32_t height,uint64_t stride,
                          std::span<const float> box,std::span<const float> intrinsics,bool decoder_operations=false);
    const std::vector<int32_t> &faces()const{return faces_;}
    const std::string &description()const{return session_.description();}
    bool bf16()const{return bf16_;}
    body_inference_options inference_options()const{return inference_;}
private:
    tensor_archive backbone_,branch_,mhr_;
    weight_map weights_;
    std::vector<int32_t> hand_indices_,faces_;
    neural_session session_;
    std::unique_ptr<dino_resident_stack> resident_;
    scalar_division arithmetic_;
    bool bf16_;
    body_inference_options inference_;
    std::mutex mutex_;
};
body_pipeline_shape trained_body_pipeline_shape(bool decoder_operations=false,body_inference_options inference={});
}
