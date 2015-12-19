#include "nmpi.h"
#include "NESTNodeNeuron.h"

class H5Neurons
{
private:
  omp_lock_t tokenLock;
  
  void CreateSubnets();
  void CreateNeurons();
  
  NESTNeuronList neurons_;
  
  std::map<int,nest::index> subnetMap_;
  
  
  bool with_scale;
  
  std::vector<double> param_facts;
  std::vector<double> param_offsets;
  
  struct ParameterValue {
    Name name;
    double value;
    ParameterValue(Name name, double value): name(name), value(value) {};
  };
  
  std::vector<ParameterValue> const_params;

  nest::index first_neuron_id;
  
public:
  H5Neurons(const Name model_name, TokenArray param_names, const Name subnet_name);
  H5Neurons(const Name model_name, TokenArray param_names, TokenArray iparam_facts, TokenArray iparam_offsets, const Name subnet_name);
  void import(const std::string& filename);
  void addConstant(std::string name, const double value);
  
  nest::index getFirstNeuronId();
  
};