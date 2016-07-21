#include <vector>
#include <deque>
//#include "nmpi.h"
#include "NESTNodeSynapse.h"
#include <map>

#include <omp.h>

//#include "H5SynMEMPedictor.h"
#include "dictdatum.h"
#include "H5SynapseLoader.h"
#include "kernels.h"

#ifndef H5Synapses_CLASS
#define H5Synapses_CLASS

enum CommunicateSynapses_Status {NOCOM,SEND, RECV, SENDRECV, UNSET};

/**
 * H5Synapses - load Synapses from HDF5 and distribute to nodes
 * 
 */

class H5Synapses
{
private:
  omp_lock_t tokenLock;
  
  std::vector<std::string> synparam_names;
  
  size_t stride_;

  kernel_combi<double> kernel;
  GIDCollectionDatum neurons;
  
  NESTSynapseList synapses_;
  
  void singleConnect(NESTNodeSynapse& synapse, nest::index synmodel_id_, nest::Node* const target_node, const nest::thread target_thread,DictionaryDatum& d, std::vector<const Token*> v_ptr, uint64_t& n_conSynapses);
  void ConnectNeurons(uint64_t& n_conSynapses);
  uint64_t threadConnectNeurons(uint64_t& n_conSynapses);
  
  void freeSynapses();
  CommunicateSynapses_Status CommunicateSynapses();
  
public:
  H5Synapses(nest::index offset, const Name synmodel_name, TokenArray hdf5_names,TokenArray synparam_names, TokenArray synparam_facts, TokenArray synparam_offset);
  ~H5Synapses();
  void import(const std::string& syn_filename, const nest::index num_syanpses_per_process=0, const nest::index last_total_synapse=0);

  void set_status( const DictionaryDatum& d );

};

#endif
