#include "objects_point_condition.hpp"
#include <iostream>
#include <stdexcept>
int main(){try{
    sam3d::point_condition_shape s;s.fusion.embed_dims={768};s.fusion.inputs={{0,1024,0,false},{0,1024,1,false}};s.fields={sam3d::point_condition_field::object,sam3d::point_condition_field::full};
    auto run=[&]{sam3d::validate_point_condition_shape(s);};run();
    auto reject=[&]{bool caught=false;try{run();}catch(const std::invalid_argument &){caught=true;}if(!caught)throw std::runtime_error("incompatible conditioner configuration accepted");};
    s.encoder.batch=2;reject();s.encoder.batch=1;s.encoder.height=128;reject();s.encoder.height=256;s.encoder.width=128;reject();s.encoder.width=256;
    s.fusion.batch=2;reject();s.fusion.batch=1;s.fusion.embed_dims={32};reject();s.fusion.embed_dims={768};
    s.fusion.inputs[0].tokens=256;reject();s.fusion.inputs[0].tokens=1024;
    s.fields.pop_back();reject();s.fields.push_back(sam3d::point_condition_field(3));reject();s.fields.back()=sam3d::point_condition_field::full;
    s.encoder.patch=0;reject();s.encoder.patch=8;s.preprocessing.point_side=0;reject();s.preprocessing.point_side=256;
    s.fusion.embed_dims.push_back(768);reject();s.fusion.embed_dims.pop_back();run();
    std::cout<<"composed point-condition shape checks passed\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
