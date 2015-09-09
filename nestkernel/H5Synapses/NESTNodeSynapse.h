#include <iostream>

#ifndef NESTNODESYNAPSE_CLASS
#define NESTNODESYNAPSE_CLASS

typedef int int32_t;
typedef long int int64_t;
typedef unsigned int uint32_t;
typedef unsigned long int uint64_t;

struct Coords
{
  double x_;
  double y_;
  double z_;
};// source_neuron_coords;*/

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
    
    double delay;
    double weight;
    double U0;
    double TauRec;
    double TauFac;
    
    
    
    //Coords source_neuron_pos_;
    
    void set(const unsigned int& source_neuron_, const unsigned int& target_neuron_);
    
    void serialize(unsigned int* buf);
    void deserialize(unsigned int* buf);
    
    void integrateOffset(const int& offset);
    
    bool operator<(const NESTNodeSynapse& rhs) const;
};



#endif