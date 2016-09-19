#include <vector>
#include <deque>
//#include "nmpi.h"

#include <map>

#include <omp.h>

//#include "H5SynMEMPedictor.h"
#include "dictdatum.h"
#include "nest_datums.h"
#include "kernels.h"

#include "SynapseList.h"
#include "h5reader.h"

#ifndef H5Synapses_CLASS
#define H5Synapses_CLASS

enum CommunicateSynapses_Status {NOCOM,SEND, RECV, SENDRECV, UNSET};

/**
 * H5Synapses - load Synapses from HDF5 and distribute to nodes
 * 
 */

using namespace nest;
using namespace h5import;

class H5Synapses
{
private:
    omp_lock_t tokenLock_;

    std::string filename_;
    std::vector< std::string > model_params_;
    std::vector< std::string > h5comp_params_;
    kernel_combi< double > kernel_;
    GIDCollectionDatum mapping_;

    long stride_;

    size_t synmodel_id_;

    uint64_t sizelimit_;
	uint64_t transfersize_;

    void singleConnect( SynapseRef synapse, nest::index synmodel_id, nest::Node* const target_node, const nest::thread target_thread, DictionaryDatum& d ,std::vector<const Token*> v_ptr, uint64_t& n_conSynapses );

    void threadConnectNeurons( SynapseList& synapses, uint64_t& n_conSynapses );

    CommunicateSynapses_Status CommunicateSynapses( SynapseList& synapses );

    void sort( SynapseList& synapses );
    void integrateMapping( SynapseList& synapses );


    void addKernel( std::string name, TokenArray params );

public:
  H5Synapses(const DictionaryDatum& din);
  ~H5Synapses();
  void import(DictionaryDatum& dout);

  void set_status( const DictionaryDatum& din );

};

#endif
