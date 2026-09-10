#include "body_pipeline.hpp"
#include "body_camera_case.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
int main(int argc,char **argv){try{
    if(argc!=2)throw std::invalid_argument("expected camera reference fixture");
    auto reject=[](auto fn){bool caught=false;try{fn();}catch(const std::invalid_argument &){caught=true;}if(!caught)throw std::runtime_error("invalid pipeline input accepted");};
    sam3d::body_pipeline_shape shape{{1,512,512,16,1280,20,5120,32,4},{{1,512,512,16,1280,1024,1,70,519,70,true,false},6,8,64,4096,128,2,128,2,1.f,true,true,false}};
    sam3d::validate_body_pipeline_shape(shape);auto s=shape;
    s.backbone.batch=2;reject([&]{sam3d::validate_body_pipeline_shape(s);});s=shape;
    s.decoder.condition.height=256;reject([&]{sam3d::validate_body_pipeline_shape(s);});s=shape;
    s.decoder.condition.context_dim=1024;reject([&]{sam3d::validate_body_pipeline_shape(s);});
    s=shape;s.decoder.hand_branch=true;reject([&]{sam3d::validate_body_pipeline_shape(s);});
    s.trained_branch=true;s.decoder.condition.hand_tokens=true;sam3d::validate_body_pipeline_shape(s);
    const auto hand_sizes=sam3d::body_pipeline_parameter_sizes(s);
    auto has=[&](const std::string &name){return std::any_of(hand_sizes.begin(),hand_sizes.end(),[&](auto &entry){return entry.first==name;});};
    if(!has("prompt_encoder.no_mask_embed.weight") || !has("bbox_embed.layers.2.weight") || !has("hand_cls_embed.weight") ||
       !has("hand_pe_layer.positional_encoding_gaussian_matrix") || !has("head_pose_hand.local_to_world_wrist") || has("head_pose.proj.layers.0.0.weight"))
        throw std::runtime_error("incorrect hand image branch parameter namespace");
    s.decoder.condition.hand_tokens=false;reject([&]{sam3d::validate_body_pipeline_shape(s);});
    std::ifstream file(argv[1]);std::string magic;uint32_t count;
    if(!(file>>magic>>count) || magic!="S3D_CAMERA_GEOMETRY_REGRESSION_V1" || count!=12)throw std::runtime_error("bad pipeline fixture");
    uint32_t cases=0;
    for(uint32_t i=0;i<count;++i){camera_case c;
        if(!(file>>c.width>>c.height>>c.side>>c.use_center>>c.padding))throw std::runtime_error("bad camera case");
        for(auto &x:c.box)if(!(file>>x))throw std::runtime_error("bad box");
        for(auto &x:c.intrinsics)if(!(file>>x))throw std::runtime_error("bad intrinsics");
        sam3d::named_floats reference;
        for(int t=0;t<5;++t){std::string key;size_t n;if(!(file>>key>>n) || n>600000 || reference.contains(key))throw std::runtime_error("bad fixture tensor");
            auto &v=reference[key];v.resize(n);for(auto &x:v)if(!(file>>x) || !std::isfinite(x))throw std::runtime_error("bad fixture value");}
        if(c.padding!=1.25f)continue; // Pipeline deliberately uses the original default.
        if(c.width>4000 || c.height>4000)throw std::runtime_error("oversized test image");
        std::vector<uint8_t> image(size_t(c.width)*c.height*3,0);
        auto run=[&]{return sam3d::body_prepare_rgb(image,c.width,c.height,uint64_t(c.width)*3,c.box,c.intrinsics,c.side,c.use_center);};
        auto result=run();
        auto near=[&](const std::vector<float> &a,const std::vector<float> &b){if(a.size()!=b.size())throw std::runtime_error("pipeline shape mismatch");for(size_t j=0;j<a.size();++j)if(std::abs(a[j]-b[j])>1e-6f)throw std::runtime_error("pipeline camera differs from original");};
        near(result.at("box_center"),reference.at("00.center"));near(result.at("box_size"),{reference.at("01.scale")[0]});
        near(result.at("affine"),reference.at("02.affine"));near(result.at("rays"),reference.at("03.rays"));near(result.at("cliff"),reference.at("04.cliff"));
        const float black[3]={-.485f/.229f,-.456f/.224f,-.406f/.225f};auto &normalized=result.at("normalized_rgb");
        if(normalized.size()!=size_t(c.side)*c.side*3)throw std::runtime_error("bad normalized layout");
        for(size_t j=0;j<normalized.size();++j)if(std::abs(normalized[j]-black[j/(c.side*c.side)])>1e-6f)throw std::runtime_error("bad black-image normalization");
        auto saved=image;image.clear();reject(run);image=std::move(saved);
        auto camera=c.intrinsics;c.intrinsics[0]=0;reject(run);c.intrinsics=camera;
        auto side=c.side;c.side=513;reject(run);c.side=side;++cases;
    }
    file>>std::ws;if(!file.eof() || cases==0)throw std::runtime_error("bad pipeline fixture coverage");
    std::cout<<cases<<" original crop/camera preparation cases and pipeline rejection checks passed; full neural composition unverified\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
