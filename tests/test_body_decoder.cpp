#include "body_decoder.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

int main(int argc,char **argv) {
    try {
        if(argc!=3) throw std::invalid_argument("expected CPU backend and decoder fixture");
        std::ifstream file(argv[2]);std::string magic;uint32_t count;
        if(!(file>>magic>>count) || magic!="S3D_DECODER_REGRESSION_V1" || count!=6) throw std::runtime_error("invalid decoder fixture header");
        auto group=[&] {
            uint32_t count;if(!(file>>count) || count>128) throw std::runtime_error("invalid fixture count");
            sam3d::named_floats values;
            for(uint32_t i=0;i<count;++i) {
                std::string name;uint64_t size;
                if(!(file>>name>>size) || size>100000 || values.contains(name)) throw std::runtime_error("invalid fixture tensor");
                auto &v=values[name];v.resize(size);
                for(auto &x:v) if(!(file>>x) || !std::isfinite(x)) throw std::runtime_error("invalid fixture float");
            }
            return values;
        };
        auto reject=[](auto f) {bool bad=false;try{f();}catch(const std::invalid_argument&){bad=true;}
            if(!bad) throw std::runtime_error("invalid decoder input accepted");};
        sam3d::neural_session session(argv[1],"CPU",0);size_t checked=0;
        for(uint32_t index=0;index<count;++index) {
            sam3d::decoder_shape s;
            if(!(file>>s.batch>>s.tokens>>s.context_tokens>>s.token_dim>>s.context_dim>>s.heads>>s.head_dim>>s.hidden>>s.repeat_pe>>s.skip_first_pe>>s.twoway))
                throw std::runtime_error("invalid decoder dimensions");
            auto input=group(),parameters=group(),reference=group();
            auto run=[&](bool all=true) {return sam3d::body_decoder_layer(session,s,input.at("tokens"),input.at("context"),input["token_pe"],input["context_pe"],input["mask"],parameters,all);};
            auto result=run();if(result.size()!=reference.size()) throw std::runtime_error("decoder tap set mismatch");
            for(auto &[name,v]:result) {
                auto &r=reference.at(name);if(r.size()!=v.size()) throw std::runtime_error("decoder tap size mismatch");
                double maximum=0,error=0,norm=0;
                for(size_t i=0;i<v.size();++i) {double delta=double(v[i])-r[i];maximum=std::max(maximum,std::abs(delta));error=std::hypot(error,delta);norm=std::hypot(norm,double(r[i]));}
                if(maximum>1e-4 || error/std::max(norm,1e-12)>2e-5) throw std::runtime_error("decoder divergence in case "+std::to_string(index)+" at "+name+" max="+std::to_string(maximum));
                ++checked;
            }
            auto plain=run(false);
            if(plain.size()!=2 || plain.at("90.tokens")!=result.at("90.tokens") || plain.at("91.context")!=result.at("91.context"))
                throw std::runtime_error("decoder observer changes output");
            auto tokens_only=sam3d::body_decoder_layer(session,s,input.at("tokens"),input.at("context"),input["token_pe"],input["context_pe"],input["mask"],parameters,false,false);
            if(tokens_only.size()!=1 || tokens_only.at("90.tokens")!=result.at("90.tokens"))
                throw std::runtime_error("skipping decoder context readback changes tokens");
            {
                sam3d::decoder_image_snapshot snapshot(session,s,input.at("context"),input["context_pe"]);
                auto resident=[&]{return sam3d::body_decoder_layer(session,s,input.at("tokens"),{},input["token_pe"],{},input["mask"],parameters,true,true,&snapshot);};
                if(resident()!=result)throw std::runtime_error("resident image changed decoder boundaries");
                const auto saved_context=input.at("context"),saved_pe=input["context_pe"];
                // Source storage is caller-owned, but the snapshot must not be.
                input["context"].assign(saved_context.size(),.25f);input["context_pe"].assign(saved_pe.size(),.5f);
                auto other=run();
                sam3d::decoder_image_snapshot next(session,s,input.at("context"),input["context_pe"]);
                auto next_result=sam3d::body_decoder_layer(session,s,input.at("tokens"),{},input["token_pe"],{},input["mask"],parameters,true,true,&next);
                if(next_result!=other || resident()!=result)throw std::runtime_error("resident decoder image lifetime/reuse error");
                input["context"]=saved_context;input["context_pe"]=saved_pe;
                reject([&]{sam3d::body_decoder_layer(session,s,input.at("tokens"),input.at("context"),input["token_pe"],{},input["mask"],parameters,true,true,&snapshot);});
                sam3d::neural_session foreign(argv[1],"CPU",0);
                reject([&]{sam3d::body_decoder_layer(foreign,s,input.at("tokens"),{},input["token_pe"],{},input["mask"],parameters,true,true,&snapshot);});
                auto wrong_shape=s;wrong_shape.context_tokens=s.context_tokens==1024?1023:s.context_tokens+1;
                reject([&]{sam3d::body_decoder_layer(session,wrong_shape,input.at("tokens"),{},input["token_pe"],{},input["mask"],parameters,true,true,&snapshot);});
                auto bad=input.at("context");bad.pop_back();
                reject([&]{sam3d::decoder_image_snapshot invalid(session,s,bad,saved_pe);});
                bad=input.at("context");bad[0]=std::numeric_limits<float>::infinity();
                reject([&]{sam3d::decoder_image_snapshot invalid(session,s,bad,saved_pe);});
                std::vector<float> bad_pe(1,0.f);
                reject([&]{sam3d::decoder_image_snapshot invalid(session,s,saved_context,bad_pe);});
                bad_pe.assign(uint64_t(s.context_tokens)*s.context_dim,std::numeric_limits<float>::quiet_NaN());
                reject([&]{sam3d::decoder_image_snapshot invalid(session,s,saved_context,bad_pe);});
            }
            auto old=s;s.heads=0;reject([&]{run();});s=old;
            s.tokens=257;reject([&]{run();});s=old;
            auto original=input.at("tokens");input["tokens"].clear();reject([&]{run();});input["tokens"]=original;
            input["tokens"][0]=std::numeric_limits<float>::infinity();reject([&]{run();});input["tokens"]=original;
            auto mask=input["mask"];input["mask"]={.5f};reject([&]{run();});
            input["mask"].assign(uint64_t(s.batch)*s.tokens,.5f);reject([&]{run();});input["mask"]=mask;
            auto xp=input["token_pe"],cp=input["context_pe"];
            input["token_pe"]={0};reject([&]{run();});input["token_pe"]=xp;
            input["context_pe"]={0};reject([&]{run();});input["context_pe"]=cp;
            if(s.repeat_pe && !cp.empty()) {input["token_pe"].clear();reject([&]{run();});input["token_pe"]=xp;}
            parameters["unexpected"]={0};reject([&]{run();});parameters.erase("unexpected");
            auto saved=parameters.begin()->second;parameters.begin()->second.push_back(0);reject([&]{run();});parameters.begin()->second=saved;
            parameters.begin()->second[0]=std::numeric_limits<float>::quiet_NaN();reject([&]{run();});parameters.begin()->second=saved;
            auto missing=parameters.extract(parameters.begin());reject([&]{run();});parameters.insert(std::move(missing));
        }
        file>>std::ws;if(!file.eof()) throw std::runtime_error("trailing decoder fixture");
        std::cout<<"Body decoder: "<<checked<<" original boundaries, observer equality and rejection cases passed\n";
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
