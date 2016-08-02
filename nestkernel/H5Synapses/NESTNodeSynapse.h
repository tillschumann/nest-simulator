#include <iostream>
#include <vector>
#include "nest_datums.h"

#ifndef NESTNODESYNAPSE_CLASS
#define NESTNODESYNAPSE_CLASS

typedef int int32_t;
typedef long int int64_t;
typedef unsigned int uint32_t;
typedef unsigned long int uint64_t;

using namespace nest;

/*class NESTNodeSynapse
{
private:
public:  
    NESTNodeSynapse();
    NESTNodeSynapse(const unsigned int& source_neuron_, const unsigned int& target_neuron_);
    ~NESTNodeSynapse();
    

    unsigned int source_neuron_;
    unsigned int target_neuron_;
    unsigned int node_id_;
    
    
    double prop_values_[5];
    
    //unsigned int num_used_prop_values_;
      
    void set(const unsigned int& source_neuron_, const unsigned int& target_neuron_);
    
    void integrateMapping(const GIDCollection& gidc);
    
    bool operator<(const NESTNodeSynapse& rhs) const;
};*/

struct NESTSynapseRef
{
    unsigned int& source_neuron_;
    unsigned int& target_neuron_;
    unsigned int& node_id_;
    
    class ParamPtr
    {
        float* ptr_;
        size_t n_;
        ParamIterator(float* ptr, const size_t& n): ptr_(ptr)
        {}
        float& operator[](const unsigned int& i)
        {
            return *(ptr_+i);
        }
        float* first()
        {
            return ptr_;
        }
        float* end()
        {
            ptr+n_;
        }
        size_t size()
        {
            return n_;
        }
    } params_;
    
    NESTSynapseRef(int& source_neurons, int& node_id, const int& num_params, char* pool_entry):
        source_neuron_(source_neurons),
        target_neuron_(*(static_cast<float*>(pool_entry))),
        node_id_(node_id),
        params_(static_cast<float*>(pool_entry)+1, num_params)
    {};
    
    int serialize(std::vector<int>& buf)
    {
        const int begin_size = buf.size();
        buf.push_back(source_neuron_);
        buf.push_back(target_neuron_);
        buf.push_back(node_id_);
        
        for (int i=0; i<params_.size(); i++)
            buf.push_back(reinterpret_cast<int>params_[i]);
        
        return buf.size()-begin_size;
    }
    void NESTNodeSynapse::deserialize(std::vector<int>& buf)
    {
        source_neuron_ = buf.pop();
        target_neuron_ = buf.pop();
        node_id_ = buf.pop();
        
        for (int i=0; i<params_.size(); i++)
            params_[i] = reinterpret_cast<float>(buf.pop());
    }
    void NESTNodeSynapse::integrateMapping(const GIDCollection& gidc)
    {
        source_neuron_ = gidc[source_neuron_];
        target_neuron_ = gidc[target_neuron_];
        
        const nest::index vp = nest::kernel().vp_manager.suggest_vp(target_neuron_);
        node_id_  = nest::kernel().mpi_manager.get_process_id(vp);
    }
    
    NESTSynapseRef& operator=(const NESTSynapseRef& r)
    {
        source_neuron_ = r.source_neuron_;
        target_neuron_ = r.target_neuron_;
        node_id_ = r.node_id_;
        
        std::copy(f.params_.begin(), f.params_.end(), params.begin());
        return *this;
    }
    
    NESTSynapseRef& swap(NESTSynapseRef& r)
    {
        //create buf object
        unsigned int source_neuron_tmp;
        unsigned int node_id_tmp;
        std::vector<char> pool_tmp(params_.size()*sizeof(float) + sizeof(int));
        NESTSynapseRef buf(source_neuron_tmp, node_id_tmp, params_.size(), pool_tmp.begin());
        
        buf = *this;
        *this = r;
        r = buf;
        
        return *this;
    }
};


class NESTSynapseList
{
//private:
  //std::vector < NESTNodeSynapse > synapses;
    
    
public:
    std::vector<unsigned int> source_neurons;
    std::vector<unsigned int> node_id_;
    std::vector<char> property_pool_;
    
    //int synmodel_id_;
    
    int num_params_;
    std::vector < std::string > prop_names;
    
    NESTNodeSynapse(const std::vector& < std::string > prop_names):
    prop_names(prop_names), num_params(prop_names.size())
    {
        //all parameters plus target neuron id
        const num_params = prop_names.size() + 1;
        property_pool.resize(num_params_*sizeof(float));
    }
  
  /*NESTSynapseRef operator[](std::size_t idx)
  {
      const int pool_idx = idx * num_params_ * sizeof(float);
      return NESTSynapseRef(source_neurons[idx], node_id_[idx], &property_pool[pool_idx]);
  };*/
  const NESTNodeSynapse& operator[](std::size_t idx)
  { 
      const int pool_idx = idx * num_params_ * sizeof(float);
      return NESTSynapseRef(source_neurons[idx], node_id_[idx], num_params_, &property_pool[pool_idx]);

  };
  void resize(const int& n)
  {
      source_neurons.resize(n);
      node_id_.resize(n);
      property_pool.resize(n*num_params_*sizeof(float));
  }
  void clear()
  {
      source_neurons.clear();
      node_id_.clear();
      property_pool.clear();
  }
  size_t size() const
  {
    return source_neurons.size();
  }
    
  /*std::vector < NESTNodeSynapse >::iterator begin()
  {
    return synapses.begin();
  }
  std::vector < NESTNodeSynapse >::const_iterator begin() const
  {
    return synapses.begin();
  }
  std::vector < NESTNodeSynapse >::iterator end()
  {
    return synapses.end();
  }
  std::vector < NESTNodeSynapse >::const_iterator end() const
  {
    return synapses.end();
  }*/
  /*void push_back(const NESTNodeSynapse& syn)
  {
    synapses.push_back(syn);
  }*/
  
    /* return number of bytes per entry
     */
};

#endif
