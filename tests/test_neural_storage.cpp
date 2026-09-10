#include "neural.hpp"
#include "dino_block.hpp"
#include "transpose.hpp"
#include "ggml.h"
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

static void require(bool value,const char *message){if(!value)throw std::runtime_error(message);}
template<class F> static void rejects(F fn){bool bad=false;try{fn();}catch(const std::invalid_argument &){bad=true;}require(bad,"invalid storage request accepted");}
int main(int argc,char **argv){try{
    if(argc!=2 && argc!=4)throw std::invalid_argument("MODULE [BACKEND DESCRIPTION]");
    sam3d::neural_session session(argv[1],argc==4?argv[2]:"CPU",0,1,argc==4?argv[3]:"");
    for(uint64_t rows:{1,31,32,33,67})for(uint64_t columns:{1,31,32,33,65}){
        std::vector<float> input(rows*columns),output(input.size()),restored(input.size());
        for(size_t i=0;i<input.size();++i)input[i]=float(i)-.5f;
        sam3d::transpose_f32(input,output,rows,columns);
        for(size_t i=0;i<rows;++i)for(size_t j=0;j<columns;++j)
            require(output[j*rows+i]==input[i*columns+j],"tiled transpose layout mismatch");
        sam3d::transpose_f32(output,restored,columns,rows);require(restored==input,"transpose round trip failed");
        rejects([&]{sam3d::transpose_f32(input,output,rows,0);});
        rejects([&]{sam3d::transpose_f32(input,output,UINT64_MAX,2);});
    }
    rejects([]{sam3d::validated_weights::checked({std::numeric_limits<float>::quiet_NaN()});});
    rejects([]{sam3d::validated_weights::checked({std::numeric_limits<float>::infinity()});});
    auto weights=sam3d::validated_weights::checked({1,2,3,4});
    sam3d::named_floats source{{"matrix",{1,2,3,4}}};
    sam3d::weight_map snapshot=source,copy=snapshot,subset;
    subset["renamed"]=snapshot.at("matrix");
    source.at("matrix")[0]=99;
    require(snapshot.at("matrix")[0]==1,"snapshot aliases caller-owned mutable storage");
    require(copy.at("matrix").data()==snapshot.at("matrix").data() &&
            subset.at("renamed").data()==snapshot.at("matrix").data(),"parameter subsets copy float data");
    source.at("matrix")[0]=std::numeric_limits<float>::infinity();
    rejects([&]{sam3d::weight_map bad=source;});
    {
        std::unique_ptr<ggml_context,decltype(&ggml_free)> c(ggml_init({ggml_tensor_overhead()*32,nullptr,true}),ggml_free);
        auto *matrix=session.parameter(c.get(),snapshot.at("matrix"),2,2);
        auto *row=session.parameter(c.get(),copy.at("matrix"),4,1);
        auto *column=session.parameter(c.get(),subset.at("renamed"),1,4);
        require(matrix->data==row->data && row->data==column->data,"parameter layouts do not share resident bytes");
        std::array<float,4> read{};ggml_backend_tensor_get(column,read.data(),0,sizeof read);
        require(read==std::array<float,4>{1,2,3,4},"parameter reshape changed values");
        rejects([&]{session.parameter(c.get(),snapshot.at("matrix"),3,2);});
        rejects([&]{session.parameter(c.get(),snapshot.at("matrix"),0,4);});
        rejects([&]{session.parameter(c.get(),snapshot.at("matrix"),UINT64_MAX,2);});
        rejects([&]{session.parameter(nullptr,snapshot.at("matrix"),2,2);});
    }
    ggml_backend_buffer_t previous=nullptr;
    {
        std::vector<float> input{1,2,3,4},kernel{1,0,0,1},bias{.25f};
        auto full=sam3d::patch_embed(session,{1,1,2,2,1,2},input,kernel,bias);
        auto compact=sam3d::patch_embed(session,{1,1,2,2,1,2},input,kernel,bias,false);
        require(compact.patches.empty() && compact.projection.empty() && compact.tokens==full.tokens,
                "patch diagnostic readbacks change output");
    }
    for(int iteration=0;iteration<4;++iteration){
        std::unique_ptr<ggml_context,decltype(&ggml_free)> c(ggml_init({ggml_tensor_overhead()*32+ggml_graph_overhead_custom(32,false),nullptr,true}),ggml_free);
        auto *w=session.constant(c.get(),weights,2,2);
        rejects([&]{session.constant(c.get(),weights,4,1);});
        rejects([&]{session.constant(c.get(),weights,0,4);});
        auto *x=ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,2);
        auto *y=ggml_mul_mat(c.get(),w,x);
        auto *g=ggml_new_graph_custom(c.get(),32,false);ggml_build_forward_expand(g,y);
        auto scratch=session.allocate(c.get());
        if(previous)require(previous==scratch.get(),"same-size scratch not reused");
        previous=scratch.get();
        std::array<float,2> input{float(iteration+1),2},output{};
        ggml_backend_tensor_set(x,input.data(),0,sizeof input);
        require(ggml_backend_graph_compute(session.backend(),g)==GGML_STATUS_SUCCESS,"compute failed");
        ggml_backend_tensor_get(y,output.data(),0,sizeof output);
        require(output[0]==input[0]+4 && output[1]==3*input[0]+8,"stale scratch/constant values");
        // A nested lease cannot reuse the currently live graph's buffer.
        std::unique_ptr<ggml_context,decltype(&ggml_free)> nested(ggml_init({ggml_tensor_overhead()*2,nullptr,true}),ggml_free);
        ggml_new_tensor_1d(nested.get(),GGML_TYPE_F32,1);
        auto other=session.allocate(nested.get());require(other.get()!=scratch.get(),"live scratch aliases");
    }
    std::cout<<"immutable weights, invalid values/layouts, scratch reuse and nested isolation pass\n";
    return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
