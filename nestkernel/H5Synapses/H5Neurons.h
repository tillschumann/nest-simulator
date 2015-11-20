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
  
public:
  H5Neurons(const Name model_name, TokenArray param_names, const Name subnet_name);
  void import(const std::string& filename);
  
  
};