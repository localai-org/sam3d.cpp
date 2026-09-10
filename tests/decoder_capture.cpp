#include "body_decoder.hpp"
#include <array>
#include <bit>
#include <charconv>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc,char **argv) {
    try {
        if(argc!=7 && argc!=8) throw std::invalid_argument("usage: decoder-capture MODULE CPU|Vulkan DEVICE DESCRIPTION|- INPUT OUTPUT [THREADS]");
        auto number=[](const char *t) {uint32_t v;auto [end,error]=std::from_chars(t,t+std::strlen(t),v);
            if(error!=std::errc() || *end) throw std::invalid_argument("invalid numeric argument");return v;};
        static_assert(std::endian::native==std::endian::little);
        std::ifstream file(argv[5],std::ios::binary);
        auto read=[&](void *p,size_t bytes) {
            if(!file.read(static_cast<char *>(p),bytes)) throw std::runtime_error("truncated decoder input");
        };
        std::array<char,8> magic;read(magic.data(),8);
        if(std::string(magic.data(),8)!="S3DDEC01") throw std::invalid_argument("invalid decoder input");
        std::array<uint32_t,11> dims;read(dims.data(),sizeof(dims));
        auto [b,n,m,d,c,h,dh,f,repeat,skip,twoway]=dims;
        if(repeat>1 || skip>1 || twoway>1) throw std::invalid_argument("invalid decoder flags");
        sam3d::decoder_shape s{b,n,m,d,c,h,dh,f,bool(repeat),bool(skip),bool(twoway)};sam3d::validate_decoder_shape(s);
        uint32_t count;read(&count,4);if(count>128) throw std::invalid_argument("too many decoder inputs");
        sam3d::named_floats parameters;uint64_t total=0;
        for(uint32_t i=0;i<count;++i) {
            uint32_t length;read(&length,4);if(length==0 || length>128) throw std::invalid_argument("invalid tensor name");
            std::string name(length,'\0');read(name.data(),length);
            uint64_t size;read(&size,8);total+=size;
            if(size>8*1024*1024 || total>128*1024*1024) throw std::invalid_argument("oversized decoder fixture");
            if(parameters.contains(name)) throw std::invalid_argument("duplicate decoder tensor");
            auto &v=parameters[name];v.resize(size);read(v.data(),size*4);
        }
        if(file.peek()!=std::ifstream::traits_type::eof()) throw std::invalid_argument("trailing decoder input");
        auto take=[&](const std::string &name) {auto node=parameters.extract(name);return node.empty()?std::vector<float>{}:std::move(node.mapped());};
        auto x=take("tokens"),context=take("context"),xp=take("token_pe"),cp=take("context_pe"),mask=take("mask");
        sam3d::neural_session session(argv[1],argv[2],number(argv[3]),argc==8?number(argv[7]):1,std::string(argv[4])=="-"?"":argv[4]);
        std::cerr<<"backend="<<session.description()<<'\n';
        auto taps=sam3d::body_decoder_layer(session,s,x,context,xp,cp,mask,parameters);
        std::ofstream output(argv[6],std::ios::binary);
        for(auto &[name,v]:taps) if(!output.write(reinterpret_cast<const char *>(v.data()),v.size()*4)) throw std::runtime_error("decoder output write failed");
        output.close();if(!output) throw std::runtime_error("decoder output close failed");
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
