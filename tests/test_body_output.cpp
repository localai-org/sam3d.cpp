#include "body_output.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
int main(int argc,char **argv){try{
    if(argc!=3)throw std::invalid_argument("expected CPU module and Body output fixture");std::ifstream file(argv[2]);std::string magic;uint32_t batch,vertices;
    if(!(file>>magic>>batch>>vertices) || magic!="S3D_BODY_OUTPUT_V1" || batch!=2 || vertices!=16)throw std::runtime_error("invalid Body output fixture");
    auto group=[&]{size_t count;if(!(file>>count) || count>16)throw std::runtime_error("invalid group count");sam3d::named_floats r;for(size_t i=0;i<count;++i){std::string name;size_t n;if(!(file>>name>>n) || n>100000 || r.contains(name))throw std::runtime_error("invalid tensor");auto &v=r[name];v.resize(n);for(float &x:v)if(!(file>>x) || !std::isfinite(x))throw std::runtime_error("invalid float");}return r;};
    auto input=group(),expected=group();file>>std::ws;if(!file.eof() || input.size()!=3 || expected.size()!=12)throw std::runtime_error("invalid groups/trailing data");
    sam3d::neural_session session(argv[1],"CPU",0);auto arithmetic=sam3d::scalar_division::direct;
    auto run=[&]{return sam3d::body_map_geometry(session,batch,vertices,input.at("vertices_cm"),input.at("skeleton"),input.at("mapping"),arithmetic);};auto result=run();if(result.size()!=expected.size())throw std::runtime_error("tap set mismatch");
    double worst=0;for(auto &[name,v]:result){auto &r=expected.at(name);if(v.size()!=r.size())throw std::runtime_error("tap shape mismatch");double maximum=0,error=0,norm=0;
        for(size_t i=0;i<v.size();++i){if(!std::isfinite(v[i]))throw std::runtime_error("nonfinite result");maximum=std::max(maximum,std::abs(double(v[i])-r[i]));error=std::hypot(error,double(v[i])-r[i]);norm=std::hypot(norm,double(r[i]));}
        worst=std::max(worst,maximum);if(maximum>1e-4 || error/std::max(norm,1e-12)>2e-5)throw std::runtime_error("Body output divergence at "+name);}
    // Catch normalizing quaternions or replacing RoMa's diagonal formula with
    // the unit-only shortcut. The original map is homogeneous of degree two.
    auto original=input["skeleton"];for(size_t i=0;i<batch*127;++i)for(size_t k=3;k<7;++k)input["skeleton"][i*8+k]*=2.f;auto doubled=run();
    for(size_t i=0;i<doubled.at("03.joint_rotations").size();++i)if(doubled.at("03.joint_rotations")[i]!=4.f*result.at("03.joint_rotations")[i])throw std::runtime_error("quaternion normalization/diagonal shortcut introduced");input["skeleton"]=original;
    for(auto [a,b]:{std::pair{"00.vertices_m","90.vertices"},std::pair{"01.joints_m","91.joints"},std::pair{"08.first70","92.keypoints"}})
        for(size_t i=0;i<result.at(a).size();++i)if(result.at(b)[i]!=(i%3?-result.at(a)[i]:result.at(a)[i]))throw std::runtime_error("position axis mapping mismatch");
    auto hand=sam3d::body_map_geometry(session,batch,vertices,input.at("vertices_cm"),input.at("skeleton"),input.at("mapping"),arithmetic,true);
    for(auto &[name,v]:result){const bool all=name=="07.keypoints308",selected=name=="08.first70" || name=="92.keypoints";
        for(size_t i=0;i<v.size();++i){size_t joint=(i/3)%(all?308:70);float expected=(all || selected) && (joint<21 || joint>=42)?0.f:v[i];
            if(hand.at(name)[i]!=expected)throw std::runtime_error("hand keypoint mask/order/axis mismatch: "+name);}}
    auto reject=[&]{bool bad=false;try{run();}catch(const std::invalid_argument &){bad=true;}if(!bad)throw std::runtime_error("invalid Body mapping accepted");};
    for(const char *name:{"vertices_cm","skeleton","mapping"}){auto saved=input[name];input[name].pop_back();reject();input[name]=saved;input[name][0]=std::numeric_limits<float>::quiet_NaN();reject();input[name]=saved;}
    batch=0;reject();batch=3;reject();batch=2;vertices=0;reject();vertices=18440;reject();vertices=16;arithmetic=static_cast<sam3d::scalar_division>(99);reject();
    std::cout<<"Body output mapping: 12 original boundaries, max_abs="<<worst<<", quaternion/axis/rejection checks passed\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
