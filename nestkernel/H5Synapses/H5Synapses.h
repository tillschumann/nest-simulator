#include <vector>
#include <deque>
//#include "nmpi.h"
#include "NESTNodeSynapse.h"
#include <map>

#include <omp.h>

//#include "H5SynMEMPedictor.h"
#include "dictdatum.h"
#include "nest_datums.h"
#include "H5SynapseLoader.h"
#include "kernels.h"

#ifndef H5Synapses_CLASS
#define H5Synapses_CLASS

enum CommunicateSynapses_Status {NOCOM,SEND, RECV, SENDRECV, UNSET};

/**
 * H5Synapses - load Synapses from HDF5 and distribute to nodes
 * 
 */

using namespace nest;

class H5Synapses
{
private:
  omp_lock_t tokenLock;
  
  std::vector<std::string> synparam_names;
  
  long stride_;

  kernel_combi<double> kernel;
  GIDCollectionDatum neurons;
  
  NESTSynapseList synapses_;
  
  std::string filename;

  long num_syanpses_per_process;
  long last_total_synapse;

  void singleConnect(NESTNodeSynapse& synapse, nest::index synmodel_id_, nest::Node* const target_node, const nest::thread target_thread,DictionaryDatum& d, std::vector<const Token*> v_ptr, uint64_t& n_conSynapses);
  //void ConnectNeurons(uint64_t& n_conSynapses);
  uint64_t threadConnectNeurons(uint64_t& n_conSynapses);
  
  void freeSynapses();
  CommunicateSynapses_Status CommunicateSynapses();
  

  void addKernel(std::string name, TokenArray params);

public:
  H5Synapses(const DictionaryDatum& din);
  ~H5Synapses();
  void import(DictionaryDatum& dout);

  void set_status( const DictionaryDatum& din );

};

#endif
