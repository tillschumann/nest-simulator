#include <vector>
#include <deque>
#include "nmpi.h"
#include "NESTNodeSynapse.h"
#include <map>

#include "H5SynMEMPedictor.h"

#include "H5SynapseLoader.h"



#ifndef H5Synapses_CLASS
#define H5Synapses_CLASS

//void NESTConnect(std::vector<NESTNodeSynapse>& synapses);
//void NESTCreateNeurons(const int& non);

enum CommunicateSynapses_Status {NOCOM,SEND, RECV, SENDRECV, UNSET};


/**
 * H5Synapses - load Synapses from HDF5 and distribute to nodes
 * 
 */

class H5Synapses
{
private:
  //GIDVector<char> neuron_type_;
  
  omp_lock_t tokenLock;
  
  
  //GIDVector<NESTNodeNeuron> neurons_;
  //std::map<int,nest::index> subnetMap_;
  
  nest::index synmodel_id_;
  
  uint32_t neuron_id_offset_;
  //uint32_t numberOfNeurons;
  
  TraceLogger tracelogger;
  
  H5SynMEMPredictor memPredictor;
  
  NESTSynapseList synapses_;
  
  /*struct SynapseModelProperties
  {
    nest::index synmodel_id; // NEST reference
    double min_delay; // 
    double C_delay;
    
    inline double get_delay_from_distance(const double& distance) const
    {
      const double delay = distance * C_delay;
      if (delay > min_delay)
	return delay;
      else
	return min_delay;
    }
  };
  SynapseModelProperties* synmodel_props;*/
  
  //void CreateNeurons();
  //void CreateSubnets();
  
  void singleConnect(nest::Node* const target_node, const nest::thread target_thread, uint64_t& n_conSynapses, nestio::Stopwatch::timestamp_t& connect_dur);
  
  void ConnectNeurons(uint64_t& n_conSynapses);
  void threadConnectNeurons(uint64_t& n_conSynapses);
  
  void freeSynapses();
  CommunicateSynapses_Status CommunicateSynapses();
  
public:
  H5Synapses(const Name& synmodel_name, TokenArray synparam_names);
  ~H5Synapses();
  void run(const std::string& syn_filename);
};

#endif