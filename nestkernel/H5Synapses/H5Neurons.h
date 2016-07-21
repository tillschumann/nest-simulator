//#include "nmpi.h"
#include <map>
#include <omp.h>
#include "NESTNodeNeuron.h"
#include "nest_types.h"
#include "nest_datums.h"
#include "name.h"
#include "tokenarray.h"
#include "kernels.h"

using namespace nest;

class H5Neurons
{
private:
  omp_lock_t tokenLock;
  
  GIDCollectionDatum CreateSubnets(const GIDCollectionDatum& added_neurons);
  GIDCollectionDatum CreateNeurons();
  
  NESTNeuronList neurons_;
  
  //std::map< int, nest::index > subnetMap_;
  
  //std::vector<double> param_facts;
  //std::vector<double> param_offsets;
  
  kernel_combi<float> kernel;
    
    std::string filename;
    
    std::vector< std::string > model_param_names;

  /*struct ParameterValue {
    std::string name;
    double value;
    ParameterValue(std::string name, double value): name(name), value(value) {};
  };*/
  
  //std::vector<ParameterValue> const_params;
  
public:
  H5Neurons(const DictionaryDatum& din);
  void import(DictionaryDatum& dout);
  void addKernel(const std::string& name, TokenArray params);
  
  //nest::index getFirstNeuronId();
  
};
