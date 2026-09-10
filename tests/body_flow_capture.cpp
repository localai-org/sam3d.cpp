#include "body_flow.hpp"
#include "body_pipeline.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
int main(int argc,char **argv){try{
    if(argc!=9 && argc!=10)throw std::invalid_argument("usage: body-flow-capture MODULE CPU|Vulkan DEVICE DESCRIPTION|- GGUF INPUT OUTPUT THREADS [BACKBONE_INPUT]");
    auto number=[](const char *p){uint32_t n;auto [end,e]=std::from_chars(p,p+std::strlen(p),n);if(e!=std::errc() || *end)throw std::invalid_argument("invalid number");return n;};
    static_assert(std::endian::native==std::endian::little);std::ifstream input(argv[6],std::ios::binary);auto read=[&](void *p,size_t n){if(!input.read(static_cast<char *>(p),n))throw std::invalid_argument("truncated Body flow input");};
    std::array<char,8> magic;read(magic.data(),8);const auto format=std::string(magic.data(),8);const bool hand=format=="S3DHFL01" || format=="S3DHRG01",operations=format=="S3DRGB03" || hand,trained=format=="S3DRGB02" || format=="S3DRGB03" || format=="S3DHRG01",rgb_mode=format=="S3DRGB01" || trained;
    if((format!="S3DFLW01" && !rgb_mode && !hand) || (argc==10)!=rgb_mode)throw std::invalid_argument("wrong Body flow input/arguments");
    std::array<uint32_t,24> v;read(v.data(),sizeof(v));float scale;read(&scale,4);
    for(auto i:{10,11,20,21,22,23})if(v[i]>1)throw std::invalid_argument("invalid flow flag");
    sam3d::body_flow_shape s{{v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],bool(v[10]),bool(v[11])},v[12],v[13],v[14],v[15],v[16],v[17],v[18],v[19],scale,bool(v[20]),bool(v[21]),bool(v[22])};
    s.capture_decoder_operations=operations;s.hand_branch=hand;
    sam3d::body_pipeline_shape trained_shape{{1,512,512,16,1280,20,5120,32,4},s,true};
    auto sizes=trained?sam3d::body_pipeline_parameter_sizes(trained_shape):sam3d::body_flow_parameter_sizes(s);std::sort(sizes.begin(),sizes.end());auto c=s.condition;const uint64_t b=c.batch;
    std::array<int32_t,54> indices;read(indices.data(),sizeof(indices));auto tensor=[&](uint64_t n){if(n>100000000)throw std::invalid_argument("oversized flow input");std::vector<float> x(n);read(x.data(),n*4);return x;};
    std::vector<int32_t> nonhand;if(hand){nonhand.resize(145);read(nonhand.data(),nonhand.size()*4);}
    std::vector<float> features,rays,cliff,prompts,previous,center,box,image_size,intrinsics,affine,crop_size;
    std::array<uint32_t,3> image_dims{};std::vector<uint8_t> pixels;
    if(rgb_mode){
        read(image_dims.data(),sizeof(image_dims));auto [iw,ih,stride]=image_dims;
        if(!iw || !ih || iw>32766 || ih>32766 || uint64_t(iw)*ih>16000000 || stride!=uint64_t(iw)*3)throw std::invalid_argument("invalid diagnostic image shape");
        box=tensor(4);intrinsics=tensor(4);pixels.resize(uint64_t(ih)*stride);read(pixels.data(),pixels.size());
        prompts=tensor(b*c.points*3);previous=tensor(v[23]?b*522:0);
    }else{
        features=tensor(b*c.context_dim*(c.height/c.patch)*(c.width/c.patch));rays=tensor(b*2*c.height*c.width);cliff=tensor(b*3);prompts=tensor(b*c.points*3);previous=tensor(v[23]?b*522:0);
        center=tensor(b*2);box=tensor(b);image_size=tensor(b*2);intrinsics=tensor(b*9);affine=tensor(b*6);crop_size=tensor(b*2);
    }
    sam3d::named_floats parameters;for(auto &[name,n]:sizes)parameters[name]=tensor(n);
    if(input.peek()!=std::ifstream::traits_type::eof())throw std::invalid_argument("trailing Body flow input");
    sam3d::tensor_archive archive(argv[5],"sam3d.mhr.lod1",sam3d::mhr_lod1_shapes());sam3d::neural_session session(argv[1],argv[2],number(argv[3]),number(argv[8]),std::string(argv[4])=="-"?"":argv[4]);std::cerr<<"backend="<<session.description()<<", layers="<<s.depth<<'\n';
    auto arithmetic=std::string(argv[2])=="Vulkan"?sam3d::scalar_division::reciprocal_multiply:sam3d::scalar_division::direct;
    sam3d::named_floats result;
    if(trained){
        sam3d::tensor_archive backbone(argv[9],"sam3d.body.dinov3.vith16plus",sam3d::body_dino_shapes());
        result=sam3d::body_from_rgb(session,archive,trained_shape,pixels,image_dims[0],image_dims[1],image_dims[2],box,intrinsics,prompts,previous,
            [&](const std::string &name,uint64_t count){return backbone.read(name,count*4);},indices,parameters,arithmetic,nonhand);
    }else if(rgb_mode){
        std::ifstream weights(argv[9],std::ios::binary);std::array<char,8> tag;std::array<uint32_t,9> dims;
        if(!weights.read(tag.data(),8) || std::string(tag.data(),8)!="S3DBBN01" || !weights.read(reinterpret_cast<char *>(dims.data()),sizeof(dims)))throw std::invalid_argument("invalid backbone header");
        auto [bb,bh,bw,bp,bd,heads,hidden,depth,storage]=dims;sam3d::backbone_shape bs{bb,bh,bw,bp,bd,heads,hidden,depth,storage};
        sam3d::body_pipeline_shape ps{bs,s};sam3d::validate_body_pipeline_shape(ps);auto order=sam3d::backbone_parameter_sizes(bs);
        // Skip the old standalone fixture's image, use ONLY its model state.
        weights.seekg(uint64_t(bb)*3*bh*bw*4,std::ios::cur);size_t cursor=0;
        result=sam3d::body_from_rgb(session,archive,ps,pixels,image_dims[0],image_dims[1],image_dims[2],box,intrinsics,prompts,previous,
            [&](const std::string &name,uint64_t count){
                if(cursor>=order.size() || order[cursor]!=std::pair{name,count})throw std::invalid_argument("backbone parameter order mismatch");++cursor;
                std::vector<float> value(count);if(!weights.read(reinterpret_cast<char *>(value.data()),count*4))throw std::invalid_argument("truncated backbone weights");
                if(name.ends_with(".norm1.weight") || name=="norm.weight")std::cerr<<"loading "<<name<<'\n';return value;
            },indices,parameters,arithmetic);
        if(cursor!=order.size() || weights.peek()!=std::ifstream::traits_type::eof())throw std::invalid_argument("trailing backbone state");
    }else result=sam3d::body_forward_decoder(session,archive,s,features,rays,cliff,prompts,previous,center,box,image_size,intrinsics,affine,crop_size,indices,parameters,arithmetic,nonhand);
    std::ofstream output(argv[7],std::ios::binary);
    for(auto &[name,x]:result)if(!output.write(reinterpret_cast<const char *>(x.data()),x.size()*4))throw std::runtime_error("Body flow write failed");output.close();if(!output)throw std::runtime_error("Body flow close failed");
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
