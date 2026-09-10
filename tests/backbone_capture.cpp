#include "dino_backbone.hpp"
#include <array>
#include <bit>
#include <charconv>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

// Diagnostic protocol only; arbitrary trained model files use checked GGUF.
// Parameters stream in validated model order, bounding host residency per block.
int main(int argc,char **argv) {
    try {
        bool bf16=false,resident_mode=false;
        while(argc>1){
            const std::string_view flag(argv[argc-1]);
            if(flag=="--bf16")bf16=true;
            else if(flag=="--resident")resident_mode=true;
            else break;
            --argc;
        }
        if (argc<7 || argc>10) throw std::invalid_argument("usage: backbone-capture MODULE CPU|Vulkan DEVICE DESCRIPTION|- INPUT OUTPUT [THREADS [GGUF [TRACE_DIRECTORY]]]");
        if(resident_mode && argc!=9)throw std::invalid_argument("--resident requires GGUF and excludes intra-block operation tracing");
        uint32_t index;
        auto [end,ec]=std::from_chars(argv[3],argv[3]+std::strlen(argv[3]),index);
        if (ec!=std::errc() || *end) throw std::invalid_argument("invalid device index");
        uint32_t threads=1;
        if (argc>=8) {
            auto [last,error]=std::from_chars(argv[7],argv[7]+std::strlen(argv[7]),threads);
            if (error!=std::errc() || *last) throw std::invalid_argument("invalid threads");
        }
        static_assert(std::endian::native==std::endian::little);
        std::ifstream in(argv[5],std::ios::binary);
        auto read=[&](void *p,size_t bytes) {
            if (!in.read(static_cast<char *>(p),bytes)) throw std::runtime_error("truncated backbone input");
        };
        std::array<char,8> magic; read(magic.data(),8);
        if (std::string(magic.data(),8)!="S3DBBN01") throw std::invalid_argument("invalid backbone input");
        std::array<uint32_t,9> dims; read(dims.data(),sizeof(dims));
        auto [b,h,w,patch,d,heads,hidden,depth,storage]=dims;
        sam3d::backbone_shape s{b,h,w,patch,d,heads,hidden,depth,storage};
        auto sizes=sam3d::backbone_parameter_sizes(s);
        std::vector<float> image(uint64_t(b)*3*h*w); read(image.data(),image.size()*4);
        sam3d::neural_session session(argv[1],argv[2],index,threads,std::string(argv[4])=="-"?"":argv[4]);
        std::cerr<<"backend="<<session.description()<<'\n';
        std::ofstream output(argv[6],std::ios::binary);
        size_t cursor=0;
        const auto observe=[&](const std::string &name,std::span<const float> v) {
            std::cerr<<"completed "<<name<<'\n';
            if (!output.write(reinterpret_cast<const char *>(v.data()),v.size_bytes()))
                throw std::runtime_error("backbone output write failed");
        };
        sam3d::backbone_block_observer block_observer;
        std::ofstream trace;
        if (argc==10) {
            const std::filesystem::path directory(argv[9]);
            if (std::filesystem::exists(directory)) throw std::invalid_argument("trace directory already exists");
            std::filesystem::create_directories(directory);
            block_observer=[&,directory](uint32_t index,const std::string &name,std::span<const float> values) {
                if (name=="00.rope_sin") {
                    if (trace.is_open()) { trace.close(); if (!trace) throw std::runtime_error("trace close failed"); }
                    trace.open(directory/("block."+std::to_string(index)+".bin"),std::ios::binary);
                }
                if (!trace.write(reinterpret_cast<const char *>(values.data()),values.size_bytes()))
                    throw std::runtime_error("block trace write failed");
            };
        }
        if (argc>=9) {
            if (dims!=std::array<uint32_t,9>{1,512,512,16,1280,20,5120,32,4})
                throw std::invalid_argument("GGUF backbone requires official Body shape");
            sam3d::tensor_archive archive(argv[8],"sam3d.body.dinov3.vith16plus",sam3d::body_dino_shapes());
            const sam3d::parameter_reader provider=[&](const std::string &name,uint64_t count){return archive.load(name,count*4);};
            std::unique_ptr<sam3d::dino_resident_stack> resident;
            if(resident_mode)resident=std::make_unique<sam3d::dino_resident_stack>(session,s,provider,bf16);
            sam3d::dino_backbone(session,s,image,provider,observe,block_observer,resident.get(),bf16);
            cursor=sizes.size();
        } else sam3d::dino_backbone(session,s,image,[&](const std::string &name,uint64_t count) {
            if (cursor>=sizes.size() || sizes[cursor]!=std::pair{name,count})
                throw std::runtime_error("backbone parameter stream order mismatch");
            ++cursor;
            std::vector<float> v(count); read(v.data(),v.size()*4); return v;
        },observe,{},nullptr,bf16);
        if (cursor!=sizes.size() || in.peek()!=std::ifstream::traits_type::eof())
            throw std::invalid_argument("trailing backbone input");
        output.close(); if (!output) throw std::runtime_error("backbone output close failed");
        if (trace.is_open()) { trace.close(); if (!trace) throw std::runtime_error("trace close failed"); }
    } catch (const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}
