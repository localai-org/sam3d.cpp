#include "objects_point_condition.hpp"
#include <array>
#include <bit>
#include <charconv>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
int main(int argc,char **argv){try{
    if(argc!=10 && argc!=11)throw std::invalid_argument("usage: point-condition-capture MODULE CPU|Vulkan DEVICE DESCRIPTION|- RAW_INPUT WEIGHTS OUTPUT_DIRECTORY THREADS CHUNK [EXPECTED_FINAL]");
    auto number=[](const char *p){uint32_t n;auto [e,ec]=std::from_chars(p,p+std::strlen(p),n);if(ec!=std::errc() || *e)throw std::invalid_argument("invalid numeric argument");return n;};
    static_assert(std::endian::native==std::endian::little);std::ifstream in(argv[5],std::ios::binary),weights(argv[6],std::ios::binary);
    auto read=[](std::ifstream &f,void *p,size_t n){if(!f.read(static_cast<char *>(p),n))throw std::invalid_argument("truncated point condition input");};
    std::array<char,8>tag;std::array<uint32_t,9>a;std::array<double,2>crop;read(in,tag.data(),8);read(in,a.data(),sizeof(a));read(in,crop.data(),sizeof(crop));
    if(std::string(tag.data(),8)!="S3DOPP01" || !a[0] || !a[1] || a[0]>4096 || a[1]>4096 || a[2]<a[0]*4 || a[2]>a[0]*4+4096 || !a[3] || !a[4] || a[3]>2048 || a[4]>2048 || a[7]>1 || a[8]>1)throw std::invalid_argument("invalid preprocessing input");
    sam3d::point_condition_shape s;auto &o=s.preprocessing;o.image_side=a[5];o.point_side=a[6];o.normalize=a[7];o.point_nan_padding=a[8];o.box_factor=crop[0];o.padding=crop[1];
    for(auto *opt:{&o.object_normalizer,&o.full_normalizer}){std::array<uint32_t,3>b;std::array<double,4>d;read(in,b.data(),sizeof(b));read(in,d.data(),sizeof(d));
        if(b[1]>1 || b[2]>1)throw std::invalid_argument("invalid normalizer flags");opt->mode=sam3d::objects_ssi_mode(b[0]);opt->allow_override=b[1];opt->raise_on_no_valid_points=b[2];opt->quantile_drop=d[0];opt->clip=d[1];opt->scale_factor=d[2];opt->log_disparity_shift=d[3];}
    sam3d::validate_objects_preprocess_options(o);
    std::vector<uint8_t> rgba(uint64_t(a[1])*a[2]);std::vector<float> xyz(uint64_t(3)*a[3]*a[4]);read(in,rgba.data(),rgba.size());read(in,xyz.data(),xyz.size()*4);
    if(in.peek()!=std::ifstream::traits_type::eof())throw std::invalid_argument("trailing raw input");
    std::array<uint32_t,5>p;read(weights,tag.data(),8);read(weights,p.data(),sizeof(p));if(std::string(tag.data(),8)!="S3DPCW01" || p[3]>1)throw std::invalid_argument("invalid point conditioner weights");
    s.encoder={1,o.point_side,o.point_side,p[0],p[1],p[2],sam3d::point_remapping(p[4]),false,false};sam3d::validate_pointpatch_shape(s.encoder);
    uint32_t n=(p[0]/p[1])*(p[0]/p[1]);s.fusion.embed_dims={p[2]};s.fusion.inputs={{0,n,0,false},{0,n,1,bool(p[3])}};s.fields={sam3d::point_condition_field::object,sam3d::point_condition_field::full};sam3d::validate_point_condition_shape(s);
    auto params=[&](const auto &sizes){sam3d::named_floats result;for(auto &[key,n]:sizes){auto &v=result[key];v.resize(n);read(weights,v.data(),n*4);}return result;};
    auto ep=params(sam3d::pointpatch_parameter_sizes(s.encoder)),fp=params(sam3d::fuser_parameter_sizes(s.fusion));
    if(weights.peek()!=std::ifstream::traits_type::eof())throw std::invalid_argument("trailing point weights");
    sam3d::neural_session session(argv[1],argv[2],number(argv[3]),number(argv[8]),std::string(argv[4])=="-"?"":argv[4]);std::cerr<<session.description()<<'\n';
    std::filesystem::path dir(argv[7]);std::filesystem::create_directories(dir);
    const std::set<std::string> encoder_taps={"11.norm1","12.qkv","19.attn_projected","21.norm2","22.fc1","23.gelu","24.fc2","30.block_output","90.output"};
    struct file{std::ofstream out;uint64_t next=0,total=0;};std::map<std::string,file> files;
    auto capture=[&](const std::string &key,std::span<const float> v,uint64_t offset,uint64_t total){
        if(key.starts_with("10.encoder.") && !encoder_taps.contains(key.substr(key.find('.',11)+1)))return;
        auto &f=files[key];if(!f.out.is_open()){f.out.open(dir/(key+".bin"),std::ios::binary);f.total=total;}
        if(offset!=f.next || total!=f.total || v.size()>total-offset)throw std::runtime_error("overlapping/incomplete conditioner taps");
        if(!f.out.write(reinterpret_cast<const char *>(v.data()),v.size()*4))throw std::runtime_error("conditioner capture write failed");f.next+=v.size();
    };
    auto result=sam3d::objects_point_condition(session,s,rgba,a[0],a[1],a[2],xyz,a[3],a[4],ep,fp,capture,number(argv[9]));
    if(argc==11){
        auto compact=sam3d::objects_point_condition(session,s,rgba,a[0],a[1],a[2],xyz,a[3],a[4],ep,fp,{},number(argv[9]));
        if(compact.conditioning!=result.conditioning || compact.embeddings!=result.embeddings)throw std::runtime_error("composed observer changed neural output");
        sam3d::named_floats actual;for(auto &[key,v]:result.prepared)actual["00.prepared."+key]=v;
        actual["90.result"]=result.conditioning;for(size_t i=0;i<result.embeddings.size();++i)actual["91.embedding."+std::to_string(i)]=result.embeddings[i];
        std::ifstream expected(argv[10]);std::string magic;uint32_t count;
        if(!(expected>>magic>>count) || magic!="S3D_POINT_CONDITION_FINAL_V1" || count!=actual.size())throw std::runtime_error("invalid composed final fixture");
        for(auto &[key,v]:actual){std::string name;size_t n;if(!(expected>>name>>n) || name!=key || n!=v.size())throw std::runtime_error("composed final fixture extent mismatch");double mx=0,e2=0,r2=0;
            for(float x:v){std::string token;if(!(expected>>token))throw std::runtime_error("truncated final fixture");float y;auto [end,ec]=std::from_chars(token.data(),token.data()+token.size(),y);if(ec!=std::errc() || end!=token.data()+token.size())throw std::runtime_error("invalid final fixture float");
                bool point=key.find("pointmap")!=std::string::npos && !key.ends_with("scale") && !key.ends_with("shift");
                if(!std::isfinite(x) || !std::isfinite(y)){if(!point || !(std::isnan(x) && std::isnan(y)))throw std::runtime_error("nonfinite composed final mismatch");continue;}
                double delta=double(x)-y;mx=std::max(mx,std::abs(delta));e2+=delta*delta;r2+=double(y)*y;
            }
            const bool mask=key.find("mask")!=std::string::npos;
            if(mx>(mask?0:1e-4) || std::sqrt(e2)/std::max(std::sqrt(r2),1e-12)>(mask?0:2e-5))throw std::runtime_error("composed original final mismatch "+key);
        }
        expected>>std::ws;if(!expected.eof())throw std::runtime_error("trailing composed final fixture");
        if(s.encoder.remap==sam3d::point_remapping::sinh){auto invalid=s;invalid.encoder.remap=sam3d::point_remapping::exp;bool caught=false;try{sam3d::objects_point_condition(session,invalid,rgba,a[0],a[1],a[2],xyz,a[3],a[4],ep,fp);}catch(const std::invalid_argument &){caught=true;}if(!caught)throw std::runtime_error("invalid original remapper domain accepted");}
        std::cerr<<count<<" original composed final fields passed\n";
    }
    capture("90.result",result.conditioning,0,result.conditioning.size());
    for(size_t i=0;i<result.embeddings.size();++i)capture("91.embedding."+std::to_string(i),result.embeddings[i],0,result.embeddings[i].size());
    for(auto &[key,f]:files){f.out.close();if(!f.out || f.next!=f.total)throw std::runtime_error("incomplete conditioner output");}
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
