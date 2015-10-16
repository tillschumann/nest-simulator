#include <iostream>
#include <vector>

#ifndef NESTNODESYNAPSE_CLASS
#define NESTNODESYNAPSE_CLASS

typedef int int32_t;
typedef long int int64_t;
typedef unsigned int uint32_t;
typedef unsigned long int uint64_t;

class NESTNodeSynapse
{
private:
public:  
    NESTNodeSynapse();
    NESTNodeSynapse(const unsigned int& source_neuron_, const unsigned int& target_neuron_);
    ~NESTNodeSynapse();
    

    unsigned int source_neuron_;
    unsigned int target_neuron_;
    unsigned int node_id_;
    
    
    double prop_values_[10];
    
    unsigned int num_used_prop_values_;
      
    void set(const unsigned int& source_neuron_, const unsigned int& target_neuron_);
    
    void serialize(unsigned int* buf);
    void deserialize(unsigned int* buf);
    
    void integrateOffset(const int& offset);
    
    bool operator<(const NESTNodeSynapse& rhs) const;
};


class NESTSynapseList
{
private:
  std::vector < NESTNodeSynapse > synapses;
  //std::vector < std::string > prop_names;
  
public:
  nest::index synmodel_id_;
  std::vector < std::string > prop_names;
  
  NESTNodeSynapse& operator[](std::size_t idx)
  {
    return synapses[idx];
  };
  const NESTNodeSynapse& operator[](std::size_t idx) const
  { 
    return synapses[idx];
  };
  void resize(const int& n)
  {
    synapses.resize(n);
  }
  void clear()
  {
    synapses.clear();
  }
  size_t size() const
  {
    return synapses.size();
  }
  std::vector < NESTNodeSynapse >::iterator begin()
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
  }
  void push_back(const NESTNodeSynapse& syn)
  {
    synapses.push_back(syn);
  }
  
  int entry_size_int()
  {
    return 3 + 5 * 2;
  }
};



#endif