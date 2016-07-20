#include <iostream>
#include <string>
#include <vector>

//#include "communicator.h"

#ifndef NESTNODENEURON_CLASS
#define NESTNODENEURON_CLASS



class NESTNodeNeuron
{
private:
public:  
    NESTNodeNeuron();
    NESTNodeNeuron(const int& subnet);
    ~NESTNodeNeuron();
    
    float parameter_values_[13];
    int subnet_;
    
    //bool operator<(const NESTNodeNeuron& rhs) const;
};

class NESTNeuronList
{  
public:
  std::vector< float > neuron_parameters_;
  std::vector< int > subsets_;
  
  nest::index model_id_;
  std::vector < std::string > parameter_names;
  
  bool with_subnet;
  std::string subnet_name;
  
  NESTNodeNeuron operator[](std::size_t idx)
  {
    NESTNodeNeuron neuron(subsets_[idx]);
    for (int i=0; i<13; i++)
      neuron.parameter_values_[i] = neuron_parameters_[idx+i];
    return neuron;
  };
  
  float getParameter(std::size_t idx, std::size_t idp)
  {
    return neuron_parameters_[idx*13+idp];
  }
  
  void setParameter(std::size_t idx, std::size_t idp, const float v)
  {
    neuron_parameters_[idx*13+idp] = v;
  }
  
  void swap(std::size_t i, std::size_t j)
  {
    float tmp[13];
    memcpy(tmp, &neuron_parameters_[i*13], sizeof(float)*13);
    memcpy(&neuron_parameters_[i*13], &neuron_parameters_[j*13], sizeof(float)*13);
    memcpy(&neuron_parameters_[j*13], tmp, sizeof(float)*13);
  }
  
  void resize(const int& n)
  {
    
    //neuron_parameters_ can be reduced; number of used entries: n/NUM_PROCESSES
    neuron_parameters_.resize(n*13);
    subsets_.resize(n);
  }
  void clear()
  {
    neuron_parameters_.clear();
    subsets_.clear();
  }
  size_t size() const
  {
    return subsets_.size();
  }
  void push_back(const NESTNodeNeuron& neuron)
  {
    for (int i=0; i<13; i++)
      neuron_parameters_.push_back(neuron.parameter_values_[i]);
  }
  int getSubnet(const std::size_t idx)
  {
    return subsets_[idx];
  }
  void setSubnet(std::size_t idx, int subnet)
  {
    subsets_[idx] = subnet;
  }
};

#endif
