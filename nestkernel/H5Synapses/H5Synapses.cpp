#include "H5Synapses.h"

#include <iostream>      
#include "nmpi.h"
#include <algorithm> 
#include <sstream>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include "timer/stopwatch.h"

#include "network.h"
#include "node.h"
#include "nestmodule.h"

#include "nest_names.h"

#include "poisson_randomdev.h"

#include "communicator.h"

#include <stdio.h>

#ifdef IS_BLUEGENE_Q
#include <spi/include/kernel/memory.h>
#endif

#include <algorithm>


#define _DEBUG_MODE 1


void H5Synapses::CreateSubnets()
{
  
  //find all subnets
  GIDVector<int> unique_subnets;
  for (int i=0; i<neurons_.size(); i++)
  {
    if (!(std::find(unique_subnets.begin(), unique_subnets.end(), neurons_[i].subnet_) != unique_subnets.end()))
    {
      unique_subnets.push_back(neurons_[i].subnet_);
    }
  }
  //del 0 subnet, because 0 subnet means main network
  int n_newSubnets = unique_subnets.size();
  if (std::find(unique_subnets.begin(), unique_subnets.end(), 0)!= unique_subnets.end())
    n_newSubnets--;
  
  //create subnets:
  const std::string sub_modname = "subnet";
  const Token sub_model = nest::NestModule::get_network().get_modeldict().lookup(sub_modname);
  const nest::index sub_model_id = static_cast<nest::index>(sub_model);  
  const long sub_last_node_id = nest::NestModule::get_network().add_node(sub_model_id, n_newSubnets);
  
  //fill subnet map with nest ids
  nest::index first_sub = sub_last_node_id - n_newSubnets+1;
  for (int i=0; i<unique_subnets.size(); i++) {
    if (unique_subnets[i]==0) {
      subnetMap_[unique_subnets[i]] = 0;
    }
    else {
      subnetMap_[unique_subnets[i]] = first_sub;
      first_sub++;
    }
    
  }
}

/*
 * 
 * 
 */
void H5Synapses::CreateNeurons()
{  
  const uint32_t non = neurons_.size();
  
  CreateSubnets();
 
  if (memPredictor.preNESTCreate(non)==0)
  {
    const std::string modname = "aeif_cond_exp";
    const Token model = nest::NestModule::get_network().get_modeldict().lookup(modname);      
    // create
    const nest::index model_id = static_cast<nest::index>(model);
    
    //jump to main network
    nest::index current_subnet=0;
    nest::NestModule::get_network().go_to(current_subnet);
  
    int last_index=0;
    for (int i=0;i<non;i++) {
      if (current_subnet!=neurons_[i].subnet_) {
	if (i>last_index) // only the case if first neuron is not 0 subnet
	  nest::NestModule::get_network().add_node(model_id, i-last_index);
	current_subnet=neurons_[i].subnet_;
	last_index=i;
	
	//jump to subnetwork
	nest::NestModule::get_network().go_to(subnetMap_[neurons_[i].subnet_]);
      }
    }    
    nest::index neuron_id = nest::NestModule::get_network().add_node(model_id, non-last_index);
    
    
    //nest::index neuron_id = nest::NestModule::get_network().add_node(model_id, non);
    
    neurons_.setOffset(-1*(neuron_id-non+1));
    
    DictionaryDatum d = DictionaryDatum( new Dictionary );
    def< double >( d, nest::names::C_m, 0. );
    def< double >( d, nest::names::Delta_T, 0. );
    def< double >( d, nest::names::E_L, 0. );
    def< double >( d, nest::names::E_ex, 0. );
    def< double >( d, nest::names::E_in, 0. );
    def< double >( d, nest::names::V_peak, 0. );
    def< double >( d, nest::names::V_reset, 0. );
    def< double >( d, nest::names::V_th, 0. );
    def< double >( d, nest::names::a, 0. );
    def< double >( d, nest::names::b, 0. );
    
    //not give:
    //def< double >( d, names::t_ref, t_ref_ );
    //def< double >( d, names::g_L, g_L );
    //def< double >( d, names::tau_syn_ex, tau_syn_ex );
    //def< double >( d, names::tau_syn_in, tau_syn_in );
    //def< double >( d, names::tau_w, tau_w );
    //def< double >( d, names::I_e, I_e );
    //def< double >( d, names::gsl_error_tol, gsl_error_tol );

    for (int i=0;i<non;i++) {
      nest::Node* node = nest::NestModule::get_network().get_node(neuron_id);
      if (nest::NestModule::get_network().is_local_node(node))
      {
	( *d )[ nest::names::C_m ] = static_cast<double>(neurons_[neuron_id].C_m_);
	( *d )[ nest::names::Delta_T ] = static_cast<double>(neurons_[neuron_id].Delta_T_);
	( *d )[ nest::names::E_L ] = static_cast<double>(neurons_[neuron_id].E_L_);
	( *d )[ nest::names::E_ex ] = static_cast<double>(neurons_[neuron_id].E_ex_);
	( *d )[ nest::names::E_in ] = static_cast<double>(neurons_[neuron_id].E_in_);
	( *d )[ nest::names::V_peak ] = static_cast<double>(neurons_[neuron_id].V_peak_);
	( *d )[ nest::names::V_reset ] = static_cast<double>(neurons_[neuron_id].V_reset_);
	( *d )[ nest::names::V_th ] = static_cast<double>(neurons_[neuron_id].V_th_);
	( *d )[ nest::names::a ] = static_cast<double>(neurons_[neuron_id].a_);
	( *d )[ nest::names::b ] = static_cast<double>(neurons_[neuron_id].b_);
	node->set_status(d);
      }
      neuron_id--;
    }
      
    std::cout << "CreateNeurons \trank= " << nest::Communicator::get_rank() << "\n";  
  }
}

void H5Synapses::singleConnect(const NESTNodeSynapse& synapse, nest::Node* const target_node, const nest::thread target_thread, uint64_t& n_conSynapses, nestio::Stopwatch::timestamp_t& connect_dur)
{
  nest::index source = synapse.source_neuron_;
  
  
  // check whether the target is on this process
  if (nest::NestModule::get_network().is_local_node(target_node))
  {
    // calculate delay of synapse:

    DictionaryDatum d = DictionaryDatum( new Dictionary );
    
    def< double_t >( d, nest::names::delay, synapse.delay );
    def< double_t >( d, nest::names::weight, synapse.weight );
    //def< double_t >( d, nest::names::dU, U_ );
    def< double_t >( d, nest::names::dU, synapse.U0 );
    def< double_t >( d, nest::names::tau_rec, synapse.TauRec );
    def< double_t >( d, nest::names::tau_fac, synapse.TauFac );
    //def< double_t >( d, nest::names::x, x_ );
    
    //nest::index synmodel_id;
    
    //set synapse type and check for delay boundary
    
    // current selection of synapse is based on source neuron
    SynapseModelProperties& synmodel_prop = synmodel_props[0]; //
	
    //if (target_thread != section_ptr)
      //std::cout << "ConnectNeurons thread Ouups!!" << "\n";
    
    nestio::Stopwatch::timestamp_t begin= nestio::Stopwatch::get_timestamp();
  
    
    bool success = nest::NestModule::get_network().connect(source, target_node->get_gid(), d, synmodel_prop.synmodel_id);    
    
    //( *d )[ nest::names::delay ] = synapse.delay;
    
    begin = nestio::Stopwatch::get_timestamp() - begin;
    if (success)
      n_conSynapses++;
    if (begin > 0)
      connect_dur+= begin;
  }
  else
  {
    std::cout << "singleConnect Ouups!!" << "\n";
  }
}

void H5Synapses::threadConnectNeurons(const std::deque<NESTNodeSynapse>& synapses, uint64_t& n_conSynapses)
{
  const int& num_processes = nest::Communicator::get_num_processes();
  const int& num_vp = nest::Communicator::get_num_virtual_processes(); 
  
  uint64_t n_conSynapses_tmp=0;
  
  if (memPredictor.preNESTConnect(synapses.size())==0)
  {
    
    #pragma omp parallel default(shared) reduction(+:n_conSynapses_tmp)
    {
      nestio::Stopwatch::timestamp_t connect_dur=0;
      nestio::Stopwatch::timestamp_t before_connect=nestio::Stopwatch::get_timestamp();
      
      
      const int tid = nest::NestModule::get_network().get_thread_id();
      
      if (num_vp != (int)(num_processes*omp_get_num_threads()))
	std::cout << "ERROR: NEST threads " << num_vp << " are not equal to OMP threads " << omp_get_num_threads() << "\n";
    
      //without preprocessing:
      //only connect neurons which are on local thread otherwise skip
      
      for (int i=0;i<synapses.size();i++) {
	
	const nest::index target = synapses[i].target_neuron_;
	
	nest::Node* const target_node = nest::NestModule::get_network().get_node(target);
	const nest::thread target_thread = target_node->get_thread();
	
	if (target_thread == tid)  // ((synapses[i].target_neuron_ % num_vp) / num_processes == section_ptr) // synapse belongs to local thread, connect function is thread safe for this condition
	{
	  singleConnect(synapses[i], target_node, target_thread, n_conSynapses_tmp, connect_dur);
	}
      }
      tracelogger.store(tid,"nest::connect", before_connect, connect_dur);
    }
    std::stringstream ss;
    ss << "threadConnectNeurons new_cons=" << synapses.size();
    TraceLogger::print_mem(ss.str());  
  }
  
  n_conSynapses += n_conSynapses_tmp;
}

void H5Synapses::ConnectNeurons(const std::deque<NESTNodeSynapse>& synapses, uint64_t& n_conSynapses)
{
  int num_processes = nest::Communicator::get_num_processes();  
  
  nestio::Stopwatch::timestamp_t connect_dur=0;
  nestio::Stopwatch::timestamp_t before_connect=nestio::Stopwatch::get_timestamp();
  
  if (memPredictor.preNESTConnect(synapses.size())==0)
  {
    for (int i=0; i< synapses.size(); i++) {
      const nest::index target = synapses[i].target_neuron_;
      nest::Node* const target_node = nest::NestModule::get_network().get_node(target);
      const nest::thread target_thread = target_node->get_thread();
      
      
      singleConnect(synapses[i], target_node, target_thread, n_conSynapses, connect_dur);
    }
  
    tracelogger.store(0,"nest::connect", before_connect, connect_dur);
  }
}

/**
 * 
 */
CommunicateSynapses_Status H5Synapses::CommunicateSynapses(std::deque<NESTNodeSynapse>& synapses)
{
  uint32_t num_processes = nest::Communicator::get_num_processes();
  
  std::stringstream sstream;
  int sendcounts[num_processes], recvcounts[num_processes], rdispls[num_processes+1], sdispls[num_processes+1];
  for (int32_t i=0;i<num_processes;i++) {
    sendcounts[i]=0;
    sdispls[i]=0;
    recvcounts[i]=-999;
    rdispls[i]=-999;
  }
  
  uint32_t* send_buffer = new uint32_t[synapses.size()*13];
  
  uint32_t* ptr_send_buffer=send_buffer;
  for (uint32_t i=0; i<synapses.size(); i++) {
    NESTNodeSynapse& syn = synapses[i];
    sendcounts[syn.node_id_]+=13;
    syn.serialize(ptr_send_buffer);
    ptr_send_buffer+=13;
  }
  
  tracelogger.begin(0, "mpi wait");
  MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
  tracelogger.end(0, "mpi wait");
  
  rdispls[0] = 0;
  sdispls[0] = 0;
  for (uint32_t i=1;i<num_processes+1;i++) {
    sdispls[i] = sdispls[i-1] + sendcounts[i-1];
    rdispls[i] = rdispls[i-1] + recvcounts[i-1];
  }  
  
  const int32_t recv_synpases_count = rdispls[num_processes]/13;
  
  
  //implement check if recv counts does not fit in memory??
  uint32_t* recvbuf= new uint32_t[rdispls[num_processes]];
 
    
  MPI_Alltoallv(send_buffer, sendcounts, sdispls, MPI_UNSIGNED, recvbuf, recvcounts, rdispls, MPI_UNSIGNED, MPI_COMM_WORLD);
  delete[] send_buffer;
  
  //std::vector<NESTNodeSynapse> own_synapses(recv_synpases_count);  
  //synapses.swap(own_synapses);
  
  
  //fill deque with recevied entries
  synapses.resize(recv_synpases_count); 
  
  uint32_t* ptr_recv_buffer=recvbuf;
  for (uint32_t i=0; i<synapses.size(); i++) {
    NESTNodeSynapse& syn = synapses[i];
    syn.deserialize(ptr_recv_buffer);
    ptr_recv_buffer+=13;
  }
  delete[] recvbuf;

  if (sdispls[num_processes]>0 && rdispls[num_processes]>0)
    return SENDRECV;
  else if (sdispls[num_processes]>0)
    return SEND;
  else if (rdispls[num_processes-1]>0)
    return RECV;
  else
    return NOCOM;
}

H5Synapses::H5Synapses()
{
  //create synapse model SynapseModelProperties
  
  synmodel_props = new SynapseModelProperties[2];
  
  //for loop is comming soon ;)
  //synapses should be stored in a list - maybe from hdf5 files or from sli script - both fine
  {
    const Token synmodel = nest::NestModule::get_network().get_synapsedict().lookup("syn_in");
    synmodel_props[0].synmodel_id = static_cast<nest::index>(synmodel);
    synmodel_props[0].min_delay = 0.4;
    synmodel_props[0].C_delay = 0.001;
  }
  {
    const Token synmodel = nest::NestModule::get_network().get_synapsedict().lookup("syn_ex");
    synmodel_props[1].synmodel_id = static_cast<nest::index>(synmodel);
    synmodel_props[1].min_delay = 0.75;
    synmodel_props[1].C_delay = 0.001;
  }
}

H5Synapses::~H5Synapses()
{
  delete[] synmodel_props;
}

void H5Synapses::freeSynapses(std::deque<NESTNodeSynapse>& synapses)
{
  //std::deque<NESTNodeSynapse> empty_synapses_vec(0);
  //empty_synapses_vec.reserve(synapses.size());
  //synapses.swap(empty_synapses_vec);
  
  synapses.clear();
}

void H5Synapses::run(const std::string& con_dir, const std::string& hdf5_cell_file)
{
  int rank = nest::Communicator::get_rank();
  int size = nest::Communicator::get_num_processes();
  
  std::cout << "Start H5Synapses" << "\n";
  std::cout << "max threads=" << omp_get_max_threads() << "\n";
    
    
  TraceLogger::print_mem("NEST base"); 
  if (rank==0) {
     numberOfNeurons= HDF5Mike::getNumberOfNeurons(hdf5_cell_file.c_str());
     //HDF5Mike::loadAllNeuronCoords(hdf5_coord_file.c_str(), numberOfNeurons, neurons_pos_);
     
     HDF5Mike::loadAllNeurons(hdf5_cell_file.c_str(),numberOfNeurons, neurons_);
  }
  
  MPI_Bcast(&numberOfNeurons, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  
  
  if (rank>0)
    neurons_.resize(numberOfNeurons); 
  
  MPI_Bcast(&neurons_[0], numberOfNeurons*sizeof(NESTNodeNeuron), MPI_BYTE, 0, MPI_COMM_WORLD); 
  TraceLogger::print_mem("with neuron parameters");
  
  // Create Neurons
  CreateNeurons();
  //neurons_.setOffset(-1);
  //neuron_type_.setOffset(-1);
  
  // oberserver variables for validation
  // sum over all after alg has to be equal
  uint64_t n_readSynapses=0;
  uint64_t n_SynapsesInDatasets=0;
  uint64_t n_memSynapses=0;
  
  uint64_t n_conSynapses=0;
  
  tracelogger.begin(0,"run");
  
  CommunicateSynapses_Status com_status=UNSET;
  std::deque<NESTNodeSynapse> synapses;
  
  HDF5Mike h5Mike(con_dir,n_readSynapses,n_SynapsesInDatasets);
  
  
  uint64_t nos = 1e6; 
  
  //load datasets from files
  while (!h5Mike.endOfMikeFiles())
  {
    memPredictor.predictBestLoadNos(nos);
    
    tracelogger.begin(0,"loadSynapses");
    h5Mike.iterateOverSynapsesFromFiles(synapses, nos);      
    tracelogger.end(0,"loadSynapses");
      
    tracelogger.begin(0,"sort");
    for (int i=0; i< synapses.size(); i++)
      synapses[i].integrateOffset(-1*neurons_.offset_-1); // inverse NEST offset + csaba offset (offset to 0-indicies)
    
    std::sort(synapses.begin(), synapses.end());
    tracelogger.end(0,"sort");
    
    tracelogger.begin(0,"communicate");
    com_status = CommunicateSynapses(synapses);
    tracelogger.end(0,"communicate"); 
    
    n_memSynapses+=synapses.size();
    
    tracelogger.begin(0,"connect");
    threadConnectNeurons(synapses, n_conSynapses);
    tracelogger.end(0,"connect");
    
    freeSynapses(synapses);
  }
  
  //recieve datasets from other nodes
  //necessary because datasets may be distributed unbalanced
  while (com_status != NOCOM) {
    tracelogger.begin(0,"communicate");
    com_status = CommunicateSynapses(synapses);
    tracelogger.end(0,"communicate");
    
    n_memSynapses+=synapses.size();
    
    tracelogger.begin(0,"connect");
    threadConnectNeurons(synapses, n_conSynapses);
    tracelogger.end(0,"connect");
    
    freeSynapses(synapses);
  }
  
  tracelogger.end(0,"run");
  

  std::cout << "rank="<< rank << "\tn_readSynapses=" << n_readSynapses << "\tn_conSynapses=" << n_conSynapses << "\tn_memSynapses="<< n_memSynapses<< "\tn_SynapsesInDatasets=" << n_SynapsesInDatasets <<  "\n";
}