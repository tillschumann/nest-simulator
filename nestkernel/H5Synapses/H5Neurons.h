//#include "nmpi.h"
#include <map>
#include <omp.h>
#include "NESTNodeNeuron.h"
#include "nest_types.h"
#include "name.h"
#include "tokenarray.h"
#include "kernels.h"


class H5Neurons
{
private:
  omp_lock_t tokenLock;
  
  void CreateSubnets();
  void CreateNeurons();
  
  NESTNeuronList neurons_;
  
  std::map< int, nest::index > subnetMap_;
  
  
  bool with_scale;
  
  std::vector<double> param_facts;
  std::vector<double> param_offsets;
  
  kernel_combi<float> kernel;

  struct ParameterValue {
    std::string name;
    double value;
    ParameterValue(std::string name, double value): name(name), value(value) {};
  };
  
  std::vector<ParameterValue> const_params;

  nest::index first_neuron_id;
  
public:
  H5Neurons(const Name model_name, TokenArray param_names, const Name subnet_name);
  void import(const std::string& filename, TokenArray dataset_names);
  void addKernel(std::string name, TokenArray params);
  
  nest::index getFirstNeuronId();
  
};
