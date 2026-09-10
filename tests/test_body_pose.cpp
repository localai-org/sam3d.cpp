#include "body_pose.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
int main(int argc,char **argv){try{
    if(argc!=3)throw std::invalid_argument("expected CPU module and pose fixture");std::ifstream file(argv[2]);std::string magic;uint32_t count;
    if(!(file>>magic>>count) || magic!="S3D_POSE_REGRESSION_V1" || count!=6)throw std::runtime_error("invalid pose fixture");
    auto group=[&]{uint32_t n;if(!(file>>n) || n>128)throw std::runtime_error("invalid tensor count");sam3d::named_floats r;
        for(uint32_t i=0;i<n;++i){std::string name;uint64_t size;if(!(file>>name>>size) || size>100000 || r.contains(name))throw std::runtime_error("invalid tensor");auto &v=r[name];v.resize(size);for(auto &x:v)if(!(file>>x) || !std::isfinite(x))throw std::runtime_error("invalid float");}return r;};
    auto reject=[](auto fn){bool bad=false;try{fn();}catch(const std::invalid_argument &){bad=true;}if(!bad)throw std::runtime_error("invalid pose accepted");};
    sam3d::neural_session session(argv[1],"CPU",0);uint32_t checked=0;
    for(uint32_t i=0;i<count;++i){sam3d::pose_shape s;uint32_t initial_flag;if(!(file>>s.batch>>s.dim>>s.hidden>>s.depth>>initial_flag) || initial_flag>1)throw std::runtime_error("invalid pose shape");sam3d::validate_pose_shape(s);
        std::array<int32_t,54> indices;for(auto &v:indices)if(!(file>>v))throw std::runtime_error("missing pose indices");auto input=group(),parameters=group(),reference=group();
        auto run=[&]{return sam3d::body_pose(session,s,input.at("token"),input.at("initial"),indices,parameters);};auto result=run();if(result.size()!=reference.size())throw std::runtime_error("pose tap set mismatch");
        std::vector<float> rotation;for(uint32_t b=0;b<s.batch;++b)rotation.insert(rotation.end(),result.at("10.pred").begin()+b*519,result.at("10.pred").begin()+b*519+6);
        auto diagnostic=sam3d::body_global_rotation(s.batch,rotation);
        for(auto &[name,v]:diagnostic)if(!name.starts_with("diagnostic.") && v!=result.at(name))throw std::runtime_error("isolated diagnostic changed production rotation arithmetic");
        reject([&]{sam3d::body_global_rotation(0,rotation);});reject([&]{sam3d::body_global_rotation(65,rotation);});
        auto bad_rotation=rotation;bad_rotation.pop_back();reject([&]{sam3d::body_global_rotation(s.batch,bad_rotation);});
        bad_rotation=rotation;bad_rotation[0]=std::numeric_limits<float>::infinity();reject([&]{sam3d::body_global_rotation(s.batch,bad_rotation);});
        for(auto &[name,v]:result){auto &r=reference.at(name);if(v.size()!=r.size())throw std::runtime_error("pose tap size mismatch at "+name);double maximum=0,error=0,norm=0;
            for(size_t k=0;k<v.size();++k){double delta=double(v[k])-r[k];maximum=std::max(maximum,std::abs(delta));error=std::hypot(error,delta);norm=std::hypot(norm,double(r[k]));}
            bool exact=name.ends_with(".singular") || name=="15.global.choice" || name=="18.global.translation" || name=="27.face";
            if((exact && maximum!=0) || maximum>1e-4 || error/std::max(norm,1e-12)>2e-5)throw std::runtime_error("pose divergence case "+std::to_string(i)+" at "+name+" max="+std::to_string(maximum));++checked;}
        // Integration invariants supplement the independent original frame
        // fixture: transfer only MHR globals, then mask the assembled params.
        std::array<float,9> world{0,-1,0,1,0,0,0,0,1};std::array<float,3> wrist{.4f,.2f,-.1f},root{-.1f,.3f,.2f};
        std::array<int32_t,145> nonhand;for(size_t k=0;k<nonhand.size();++k)nonhand[k]=int32_t(k)+6;
        sam3d::hand_pose_config config{world,wrist,root,nonhand};
        auto hand_run=[&]{return sam3d::body_pose(session,s,input.at("token"),input.at("initial"),indices,parameters,&config);};auto hand=hand_run();
        for(auto &[name,v]:result)if(name!="30.full_pose" && name!="42.full_pose_hands" && name!="90.model_params" && v!=hand.at(name))throw std::runtime_error("wrist transfer changed non-MHR output: "+name);
        for(uint32_t b=0;b<s.batch;++b)for(size_t k=0;k<204;++k){
            float value=k<3?hand.at("hand.04.translation")[b*3+k]*10.f:k<6?hand.at("hand.02.euler")[b*3+k-3]:result.at("90.model_params")[b*204+k];
            if(hand.at("hand.05.unmasked_parameters")[b*204+k]!=value || hand.at("90.model_params")[b*204+k]!=((k>=6 && k<151)?0.f:value))throw std::runtime_error("hand assembly/order mismatch");
        }
        nonhand[0]=nonhand[1];reject(hand_run);nonhand[0]=6;
        config.local_to_world={};reject(hand_run);
        auto token=input["token"];input["token"].clear();reject(run);input["token"]=token;input["token"][0]=std::numeric_limits<float>::quiet_NaN();reject(run);input["token"]=token;
        auto initial=input["initial"];input["initial"]={0};reject(run);input["initial"]=initial;
        for(int bad:{-1,136,0}){auto saved=indices[0];indices[0]=bad;reject(run);indices[0]=saved;}auto saved=indices[0];indices[0]=indices[1];reject(run);indices[0]=saved;
        parameters["extra"]={0};reject(run);parameters.erase("extra");auto missing=parameters.extract(parameters.begin());reject(run);parameters.insert(std::move(missing));
        for(auto &[name,v]:parameters){auto saved=v;v.pop_back();reject(run);v=saved;v[0]=std::numeric_limits<float>::infinity();reject(run);v=saved;}
        auto shape=s;s.depth=0;reject(run);s=shape;
        for(uint32_t b=0;b<s.batch;++b){for(uint32_t k=62;k<116;++k)if(result.at("23.body.masked")[b*133+k]!=0)throw std::runtime_error("body hand mask failed");for(uint32_t k=130;k<133;++k)if(result.at("23.body.masked")[b*133+k]!=0)throw std::runtime_error("body jaw mask failed");}
    }
    file>>std::ws;if(!file.eof())throw std::runtime_error("trailing pose fixture");std::cout<<"MHR pose prefix: "<<checked<<" original boundaries plus rejection/mask checks passed\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
