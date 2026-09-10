#include "body_hand_frame.hpp"
#include <array>
#include <bit>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
int main(int argc,char **argv){try{
    static_assert(std::endian::native==std::endian::little);
    auto require=[](bool b){if(!b)throw std::runtime_error("hand frame regression failed");};
    auto reject=[&](auto f){bool bad=false;try{f();}catch(const std::invalid_argument &){bad=true;}require(bad);};
    require(argc==2);std::ifstream in(argv[1]);std::string magic;size_t cases;
    require(bool(in>>magic>>cases) && magic=="S3D_HAND_FRAME_REGRESSION_V1" && cases==2);
    for(size_t c=0;c<cases;++c){
        uint32_t b;require(bool(in>>b) && (b==2 || b==4));
        auto tensor=[&](size_t n){std::vector<float> v(n);for(auto &x:v)require(bool(in>>x) && std::isfinite(x));return v;};
        auto rotation=tensor(b*3),translation=tensor(b*3),world=tensor(9),wrist=tensor(3),root=tensor(3),parameters=tensor(b*204),points=tensor(b*924);
        std::array<int32_t,145> indices;for(auto &x:indices)require(bool(in>>x));
        auto result=sam3d::body_hand_frame(b,rotation,translation,world,wrist,root);
        result["05.masked_parameters"]=sam3d::body_hand_mask_parameters(b,parameters,indices);
        result["06.masked_keypoints"]=sam3d::body_hand_mask_keypoints(b,points);
        size_t n;require(bool(in>>n) && n==7 && result.size()==n);
        for(auto &[key,actual]:result){std::string name;size_t count;require(bool(in>>name>>count) && name==key && count==actual.size());
            if(name.starts_with("05.") || name.starts_with("06.")){
                uint64_t expected,hash=14695981039346656037ull;require(bool(in>>expected));
                for(auto byte:std::as_bytes(std::span(actual)))hash=(hash^std::to_integer<uint8_t>(byte))*1099511628211ull;
                require(hash==expected);
            }else{auto expected=tensor(count);double squared=0,reference=0,max_abs=0;
                for(size_t i=0;i<count;++i){require(std::isfinite(actual[i]));double delta=double(actual[i])-expected[i];squared+=delta*delta;reference+=double(expected[i])*expected[i];max_abs=std::max(max_abs,std::abs(delta));}
                if(max_abs>1e-4 || std::sqrt(squared)/std::max(std::sqrt(reference),1e-12)>2e-5)throw std::runtime_error("hand frame mismatch: "+name);
            }
        }
        reject([&]{sam3d::body_hand_frame(0,rotation,translation,world,wrist,root);});
        reject([&]{sam3d::body_hand_frame(65,rotation,translation,world,wrist,root);});
        reject([&]{sam3d::body_hand_frame(b,std::span(rotation).first(1),translation,world,wrist,root);});
        auto invalid=rotation;invalid[0]=std::numeric_limits<float>::quiet_NaN();
        reject([&]{sam3d::body_hand_frame(b,invalid,translation,world,wrist,root);});
        invalid[0]=std::numeric_limits<float>::max();
        reject([&]{sam3d::body_hand_frame(b,invalid,translation,world,wrist,root);});
        auto bad=indices;bad[0]=-1;reject([&]{sam3d::body_hand_mask_parameters(b,parameters,bad);});
        bad[0]=204;reject([&]{sam3d::body_hand_mask_parameters(b,parameters,bad);});
        bad[0]=bad[1];reject([&]{sam3d::body_hand_mask_parameters(b,parameters,bad);});
        reject([&]{sam3d::body_hand_mask_parameters(b,parameters,std::span(indices).first(144));});
        reject([&]{sam3d::body_hand_mask_keypoints(b,std::span(points).first(308));});
        points[0]=std::numeric_limits<float>::infinity();reject([&]{sam3d::body_hand_mask_keypoints(b,points);});
    }
    std::string trailing;require(!(in>>trailing));std::cout<<"Original wrist-frame geometry, exact masks and rejection tests pass\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
