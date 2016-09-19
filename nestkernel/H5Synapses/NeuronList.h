#include <iostream>
#include <string>
#include <vector>
#include <cstring>

#include "nest_types.h"

//#include "communicator.h"

#ifndef NESTNODENEURON_CLASS
#define NESTNODENEURON_CLASS



class NeuronObj
{
private:
public:  
	NeuronObj(): subnet_(0)
    {}
	NeuronObj(const int& subnet): subnet_(subnet)
    {}
    std::vector<float> params_;
    int subnet_;
};

class NeuronList
{  
private:
	size_t num_parameters_;
	std::vector< int > subsets_;
public:
  nest::index model_id_;
  std::vector< float > neuron_parameters_;
  std::vector < std::string > parameter_names;
  
  bool with_subnet;
  std::string subnet_name;

	NeuronObj operator[]( size_t idx )
	{
		assert( (idx*num_parameters_+1) <= neuron_parameters_.size());
		NeuronObj neuron(subsets_[idx]);
		neuron.params_.resize( num_parameters_ );
		for (int i=0; i<num_parameters_; i++)
			neuron.params_[i] = neuron_parameters_[idx*num_parameters_+i];
		return neuron;
	};
  
  void setParameters( const std::vector < std::string >& params )
  {
	  num_parameters_ = params.size();
	  parameter_names = params;
  }
  
  void swap( size_t i, size_t j )
  {
    float tmp[num_parameters_];
    memcpy(tmp, &neuron_parameters_[i*num_parameters_], sizeof(float)*num_parameters_);
    memcpy(&neuron_parameters_[i*num_parameters_], &neuron_parameters_[j*num_parameters_], sizeof(float)*num_parameters_);
    memcpy(&neuron_parameters_[j*num_parameters_], tmp, sizeof(float)*num_parameters_);
  }
  
  void resize(const int& n)
  {
    
    //neuron_parameters_ can be reduced; number of used entries: n/NUM_PROCESSES
    neuron_parameters_.resize(n*num_parameters_);
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
