#include "dino_backbone.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

int main(int argc,char **argv) {
    try {
        if (argc!=3 && argc!=4) throw std::invalid_argument("expected module and upstream text fixture, optional backend kind");
        std::ifstream file(argv[2]); std::string magic;
        if (!(file>>magic) || magic!="S3D_BACKBONE_REGRESSION_V1") throw std::runtime_error("invalid regression fixture");
        sam3d::backbone_shape s;
        if (!(file>>s.batch>>s.height>>s.width>>s.patch>>s.dim>>s.heads>>s.hidden>>s.depth>>s.storage))
            throw std::runtime_error("invalid regression dimensions");
        auto sizes=sam3d::backbone_parameter_sizes(s);
        auto read=[&](const std::string &expected,uint64_t size) {
            std::string name; uint64_t count;
            if (!(file>>name>>count) || name!=expected || count!=size) throw std::runtime_error("regression tensor mismatch");
            std::vector<float> v(size);
            for (auto &x:v) if (!(file>>x) || !std::isfinite(x)) throw std::runtime_error("invalid regression value");
            return v;
        };
        auto image=read("image",uint64_t(s.batch)*3*s.height*s.width);
        sam3d::named_floats parameters;
        for (auto &[name,size]:sizes) parameters[name]=read(name,size);
        sam3d::neural_session session(argv[1],argc==4?argv[3]:"CPU",0);
        auto provider=[&](const std::string &name,uint64_t){return parameters.at(name);};
        size_t checks=0; std::vector<float> last;
        auto output=sam3d::dino_backbone(session,s,image,provider,[&](const std::string &name,std::span<const float> v) {
            auto expected=read(name,v.size()); ++checks;
            double maximum=0,error_norm=0,reference_norm=0;
            for (size_t i=0;i<v.size();++i) {
                const double error=double(v[i])-expected[i];
                maximum=std::max(maximum,std::abs(error)); error_norm=std::hypot(error_norm,error);
                reference_norm=std::hypot(reference_norm,double(expected[i]));
            }
            if (maximum>1e-4 || error_norm/std::max(reference_norm,1e-12)>2e-5)
                throw std::runtime_error("original backbone regression failed at "+name);
            if (name=="04.features") last.assign(v.begin(),v.end());
        });
        file>>std::ws;
        if (!file.eof() || checks!=s.depth+4 || output!=last) throw std::runtime_error("incomplete backbone regression");
        size_t block_taps=0;
        auto traced=sam3d::dino_backbone(session,s,image,provider,{},
            [&](uint32_t index,const std::string &,std::span<const float> values) {
                if (index!=block_taps/22 || values.empty()) throw std::runtime_error("invalid block trace order/data");
                ++block_taps;
            });
        if (traced!=output || block_taps!=s.depth*22)
            throw std::runtime_error("block observation changed result or lost coverage");
        sam3d::dino_resident_stack resident(session,s,provider);
        for(int repetition=0;repetition<3;++repetition){
            auto varied=image;
            if(repetition==1)for(float &v:varied)v=v*.73f+.17f;
            sam3d::named_floats expected,actual;
            auto collect=[](sam3d::named_floats &dest){return [&dest](const std::string &name,std::span<const float> v){dest[name]={v.begin(),v.end()};};};
            auto normal=sam3d::dino_backbone(session,s,varied,provider,collect(expected));
            auto cached=sam3d::dino_backbone(session,s,varied,provider,collect(actual),{},&resident);
            if(cached!=resident.run_features(s,expected.at("01.tokens"),{}))
                throw std::runtime_error("resident image/token graph disagreement");
            if(cached!=resident.run_image(s,varied,{}))
                throw std::runtime_error("resident F32 graph switching changed output");
            if(normal!=cached || expected!=actual){
                for(auto &[name,v]:expected)if(v!=actual.at(name)){
                    double delta=0;for(size_t j=0;j<v.size();++j)delta=std::max(delta,std::abs(double(v[j])-actual.at(name)[j]));
                    throw std::runtime_error("resident backbone changed "+name+" repetition "+std::to_string(repetition)+" max error "+std::to_string(delta));
                }
                throw std::runtime_error("resident backbone changed returned features");
            }
        }
        auto reject=[&](auto run) {
            bool rejected=false;
            try { run(); } catch (const std::invalid_argument &) { rejected=true; }
            if (!rejected) throw std::runtime_error("invalid backbone input accepted");
        };
        sam3d::dino_resident_stack low_resident(session,s,provider,true);
        for(int repetition=0;repetition<3;++repetition){
            auto varied=image;if(repetition==1)for(auto &v:varied)v=v*.73f+.17f;
            sam3d::named_floats expected,actual;
            auto collect=[](sam3d::named_floats &dest){return [&dest](const std::string &name,std::span<const float> v){dest[name]={v.begin(),v.end()};};};
            auto streamed=sam3d::dino_backbone(session,s,varied,provider,collect(expected),{},nullptr,true);
            auto cached=sam3d::dino_backbone(session,s,varied,provider,collect(actual),{},&low_resident,true);
            if(streamed!=cached || expected!=actual)throw std::runtime_error("resident BF16 backbone changed a captured boundary");
            if(cached!=low_resident.run_features(s,expected.at("01.tokens"),{}))
                throw std::runtime_error("resident BF16 image/token graph disagreement");
            if(cached!=sam3d::dino_backbone(session,s,varied,provider,{}, {},&low_resident,true))
                throw std::runtime_error("resident BF16 observer changed inference output");
        }
        auto never_read=[](const std::string &,uint64_t)->sam3d::validated_weights{throw std::runtime_error("resident inference reread immutable parameters");};
        if(sam3d::dino_backbone(session,s,image,never_read,{}, {},&resident)!=output)
            throw std::runtime_error("resident immutable snapshot changed");
        auto wrong_resident_shape=s;++wrong_resident_shape.storage;
        reject([&]{resident.run(wrong_resident_shape,{},{});});
        reject([&]{resident.run_features(wrong_resident_shape,{},{});});
        reject([&]{resident.run_image(wrong_resident_shape,image,{});});
        reject([&]{resident.run_image(s,std::span(image).first(1),{});});
        auto invalid_image=image;invalid_image[0]=std::numeric_limits<float>::quiet_NaN();
        reject([&]{resident.run_image(s,invalid_image,{});});
        if(resident.run_image(s,image,{})!=output)
            throw std::runtime_error("resident image input rejection corrupted following request");
        // Empty/short/long storage prefixes and both supported batch sizes.
        // Compare the full image route to the independent streamed path.
        for(uint32_t batch:{1u,2u})for(uint32_t storage:{0u,1u,7u})for(bool bf16:{false,true}){
            auto shape=s;shape.batch=batch;shape.storage=storage;
            auto varied_provider=[&](const std::string &name,uint64_t count)->sam3d::validated_weights{
                if(name=="storage_tokens"){
                    std::vector<float> values(count);const auto &base=parameters.at(name);
                    for(size_t i=0;i<values.size();++i)values[i]=base[i%base.size()];
                    return values;
                }
                return provider(name,count);
            };
            auto pixels=std::span<const float>(image).first(uint64_t(batch)*3*s.height*s.width);
            sam3d::dino_resident_stack varied(session,shape,varied_provider,bf16);
            auto expected=sam3d::dino_backbone(session,shape,pixels,varied_provider,{}, {},nullptr,bf16);
            if(varied.run_image(shape,pixels,{})!=expected || varied.run_image(shape,pixels,{})!=expected)
                throw std::runtime_error("resident image batch/storage prefix layout changed");
        }
        reject([&]{resident.parameter("not.a.parameter",1);});
        reject([&]{resident.parameter("norm.weight",1);});
        reject([&]{sam3d::dino_backbone(session,s,image,provider,{},[](uint32_t,const std::string &,std::span<const float>){},&resident);});
        auto bad=s; bad.patch=0; reject([&]{sam3d::validate_backbone_shape(bad);});
        bad=s; bad.width=0; reject([&]{sam3d::validate_backbone_shape(bad);});
        bad=s; bad.depth=33; reject([&]{sam3d::validate_backbone_shape(bad);});
        bad=s; bad.storage=8; reject([&]{sam3d::validate_backbone_shape(bad);});
        reject([&]{sam3d::dino_backbone(session,s,image,{});});
        reject([&]{sam3d::dino_backbone(session,s,std::span(image).first(1),provider);});
        auto good=parameters["patch_embed.proj.weight"];
        parameters["patch_embed.proj.weight"].pop_back();
        reject([&]{sam3d::dino_backbone(session,s,image,provider);});
        parameters["patch_embed.proj.weight"]=std::move(good);
        parameters["patch_embed.proj.weight"][0]=std::numeric_limits<float>::quiet_NaN();
        reject([&]{sam3d::dino_backbone(session,s,image,provider);});
        std::cout<<"Original whole-backbone contract: "<<checks<<" boundaries and rejection tests passed\n";
    } catch (const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}
