//#include <omp.h>
//#include <mpi.h>

#include <vector>
#include <deque>
#include <map>

//#include "H5SynMEMPedictor.h"
#include "dictdatum.h"
#include "nest_datums.h"
#include "kernels.h"

#include "h5import/synapse_buffer.h"
#include "h5import/h5synapsefile.h"

#ifndef H5Synapses_CLASS
#define H5Synapses_CLASS

enum CommunicateSynapses_Status {NOCOM,SEND, RECV, SENDRECV, UNSET};

/**
 * H5Synapses - load Synapses from HDF5 and distribute to nodes
 * 
 */

using namespace nest;
using namespace h5import;

class SynapseLoader
{
private:
    //omp_lock_t tokenLock_;

    std::string filename_;
    std::vector< std::string > model_params_;
    std::vector< std::string > h5comp_params_;
    kernel_combi< double > kernel_;

    GIDCollectionDatum mapping_;
    size_t synmodel_id_;

    uint32_t stride_;
    uint64_t sizelimit_;
	uint64_t transfersize_;

    void singleConnect( const SynapseRef& synapse, nest::index synmodel_id, nest::Node* target_node, nest::thread target_thread, DictionaryDatum& d ,std::vector<const Token*> v_ptr, uint64_t& n_conSynapses );

    void threadConnectNeurons( SynapseBuffer& synapses, uint64_t& n_conSynapses );

    CommunicateSynapses_Status CommunicateSynapses( SynapseBuffer& synapses );

    void sort( SynapseBuffer& synapses );
    void integrateMapping( SynapseBuffer& synapses );


    void addKernel( std::string name, TokenArray params );

public:
  SynapseLoader(const DictionaryDatum& din);
  ~SynapseLoader();

  void execute(DictionaryDatum& dout);
  void set_status( const DictionaryDatum& din );

};

#endif
