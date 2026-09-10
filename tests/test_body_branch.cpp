#include "body_pipeline.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
int main(int argc,char **argv){try{
    if(argc!=2)throw std::invalid_argument("expected CPU backend module");
    sam3d::neural_session session(argv[1],"CPU",0,1,"");
    sam3d::named_floats p;
    for(unsigned i=0;i<3;++i){const auto n="bbox_embed.layers."+std::to_string(i);
        p[n+".weight"]=i==2?std::vector<float>{1,0,0,1,1,1,-1,0}:std::vector<float>{1,0,0,1};
        p[n+".bias"]=std::vector<float>(i==2?4:2,0.f);}
    p["hand_cls_embed.weight"]={1,2,-1,1};p["hand_cls_embed.bias"]={.5f,-.5f};
    std::vector<float> input{1,-2,-3,4};auto out=sam3d::body_hand_detection(session,2,input,p);
    auto close=[](const auto &a,const std::vector<float> &b){if(a.size()!=b.size())throw std::runtime_error("head shape mismatch");for(size_t i=0;i<a.size();++i)if(std::abs(a[i]-b[i])>1e-6f)throw std::runtime_error("head arithmetic mismatch");};
    close(out.at("box.0.linear"),{1,-2,-3,4});close(out.at("box.1.linear"),{1,0,0,4});
    std::vector<float> logits{1,0,1,-1,0,4,4,0};close(out.at("box.2.linear"),logits);
    for(auto &v:logits)v=1/(1+std::exp(-v));close(out.at("hand_box"),logits);
    close(out.at("hand_logits"),{-2.5f,-3.5f,5.5f,6.5f});
    auto reject=[](auto fn){bool caught=false;try{fn();}catch(const std::invalid_argument &){caught=true;}if(!caught)throw std::runtime_error("invalid head input accepted");};
    reject([&]{sam3d::body_hand_detection(session,0,input,p);});
    input[0]=std::numeric_limits<float>::quiet_NaN();reject([&]{sam3d::body_hand_detection(session,2,input,p);});input[0]=1;
    p["hand_cls_embed.weight"].pop_back();reject([&]{sam3d::body_hand_detection(session,2,input,p);});
    std::cout<<"Body hand detection layout, ReLU, sigmoid, logits and rejection checks pass\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
