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


/*void H5Synapses::CreateSubnets()
{
  
  //find all subnets
  std::vector<int> unique_subnets;
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
}*/

/*
 * 
 * 
 */
/*void H5Synapses::CreateNeurons()
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
    
    //missing
    def< double >( d, nest::names::tau_syn_ex, 1.8 );
    def< double >( d, nest::names::tau_syn_in, 8.0 );
    
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
	//( *d )[ nest::names::C_m ] = static_cast<double>(neurons_[neuron_id].C_m_);
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
}*/

void H5Synapses::singleConnect(nest::Node* const target_node, const nest::thread target_thread, uint64_t& n_conSynapses, nestio::Stopwatch::timestamp_t& connect_dur)
{
  nest::index source = synapses_.source_neuron_;
  
  // check whether the target is on this process
  if (nest::NestModule::get_network().is_local_node(target_node))
  {
    // calculate delay of synapse:
    {
      DictionaryDatum d( new Dictionary );
      
      //new Token is not thread-safe
      //mutex easiest workaround
      omp_set_lock(&tokenLock);
      for (int i=0; synapses_.prop_names.size(); i++)
	def< double_t >( d, synapses_.prop_names[i], synapses_.prop_values_ [i] );
      omp_unset_lock(&tokenLock);
      
      //nest::index synmodel_id;
      
      //set synapse type and check for delay boundary
      
      // current selection of synapse is based on source neuron
      //SynapseModelProperties& synmodel_prop = synmodel_props[0]; //
      
      nestio::Stopwatch::timestamp_t begin= nestio::Stopwatch::get_timestamp();
    
      //d.addReference();
      bool success = nest::NestModule::get_network().connect(source, target_node->get_gid(), d, synmodel_id_);    
      
      //( *d )[ nest::names::delay ] = synapses_.delay;
      
      begin = nestio::Stopwatch::get_timestamp() - begin;
      if (success)
	n_conSynapses++;
      if (begin > 0)
	connect_dur+= begin;
      
      omp_set_lock(&tokenLock);
    }
    omp_unset_lock(&tokenLock);
  }
  else
  {
    std::cout << "singleConnect Ouups!!" << "\n";
  }
}

void H5Synapses::threadConnectNeurons(uint64_t& n_conSynapses)
{
  const int& num_processes = nest::Communicator::get_num_processes();
  const int& num_vp = nest::Communicator::get_num_virtual_processes(); 
  
  uint64_t n_conSynapses_tmp=0;
  
  omp_init_lock(&tokenLock);
  
  if (memPredictor.preNESTConnect(synapses_.size())==0)
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
      
      for (int i=0;i<synapses_.size();i++) {
	
	const nest::index target = synapses_[i].target_neuron_;
	
	//assert for smaller maximum neuron id
	
	nest::Node* const target_node = nest::NestModule::get_network().get_node(target);
	const nest::thread target_thread = target_node->get_thread();
	
	if (target_thread == tid)  // ((synapses_[i].target_neuron_ % num_vp) / num_processes == section_ptr) // synapse belongs to local thread, connect function is thread safe for this condition
	{
	  singleConnect(target_node, target_thread, n_conSynapses_tmp, connect_dur);
	}
      }
      tracelogger.store(tid,"nest::connect", before_connect, connect_dur);
    }
    std::stringstream ss;
    ss << "threadConnectNeurons new_cons=" << synapses_.size();
    TraceLogger::print_mem(ss.str());  
  }
  
  omp_destroy_lock(&tokenLock);
  
  n_conSynapses += n_conSynapses_tmp;
}

void H5Synapses::ConnectNeurons(uint64_t& n_conSynapses)
{
  int num_processes = nest::Communicator::get_num_processes();  
  
  nestio::Stopwatch::timestamp_t connect_dur=0;
  nestio::Stopwatch::timestamp_t before_connect=nestio::Stopwatch::get_timestamp();
  
  omp_init_lock(&tokenLock);
  
  if (memPredictor.preNESTConnect(synapses_.size())==0)
  {
    for (int i=0; i< synapses_.size(); i++) {
      const nest::index target = synapses_[i].target_neuron_;
      nest::Node* const target_node = nest::NestModule::get_network().get_node(target);
      const nest::thread target_thread = target_node->get_thread();
      
      
      singleConnect(target_node, target_thread, n_conSynapses, connect_dur);
    }
  
    tracelogger.store(0,"nest::connect", before_connect, connect_dur);
  }
  
  omp_destroy_lock(&tokenLock);
  
}

/**
 * 
 */
CommunicateSynapses_Status H5Synapses::CommunicateSynapses()
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
  
  uint32_t* send_buffer = new uint32_t[synapses_.size()*synapses_.entry_size_int()];
  
  uint32_t* ptr_send_buffer=send_buffer;
  for (uint32_t i=0; i<synapses_.size(); i++) {
    NESTNodeSynapse& syn = synapses_[i];
    sendcounts[syn.node_id_]+=synapses_.entry_size_int();
    syn.serialize(ptr_send_buffer);
    ptr_send_buffer+=synapses_.entry_size_int();
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
  
  const int32_t recv_synpases_count = rdispls[num_processes]/synapses_.entry_size_int();
  
  
  //implement check if recv counts does not fit in memory??
  uint32_t* recvbuf= new uint32_t[rdispls[num_processes]];
 
    
  MPI_Alltoallv(send_buffer, sendcounts, sdispls, MPI_UNSIGNED, recvbuf, recvcounts, rdispls, MPI_UNSIGNED, MPI_COMM_WORLD);
  delete[] send_buffer;
  
  //std::vector<NESTNodeSynapse> own_synapses(recv_synpases_count);  
  //synapses_.swap(own_synapses);
  
  
  //fill deque with recevied entries
  synapses_.resize(recv_synpases_count); 
  
  uint32_t* ptr_recv_buffer=recvbuf;
  for (uint32_t i=0; i<synapses_.size(); i++) {
    NESTNodeSynapse& syn = synapses_[i];
    syn.deserialize(ptr_recv_buffer);
    ptr_recv_buffer+=synapses_.entry_size_int();
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

H5Synapses::H5Synapses(const Name& synmodel_name, TokenArray synparam_names): neuron_id_offset_(0)
{
  //create synapse model SynapseModelProperties
  
  //for loop is comming soon ;)
  //synapses should be stored in a list - maybe from hdf5 files or from sli script - both fine
  
  for (int i=0; i<synparam_names.size(); i++)
    synapses_.prop_names.push_back(synparam_names[i]);

  const Token synmodel = nest::NestModule::get_network().get_synapsedict().lookup(synmodel_name);
  synapses_.synmodel_id_ = static_cast<nest::index>(synmodel);
}

H5Synapses::~H5Synapses()
{
}

void H5Synapses::freeSynapses()
{
  //std::deque<NESTNodeSynapse> empty_synapses_vec(0);
  //empty_synapses_vec.reserve(synapses.size());
  //synapses.swap(empty_synapses_vec);
  
  synapses_.clear();
}

void H5Synapses::run(const std::string& syn_filename)
{
  int rank = nest::Communicator::get_rank();
  int size = nest::Communicator::get_num_processes();
  
  std::cout << "Start H5Synapses" << "\n";
  std::cout << "max threads=" << omp_get_max_threads() << "\n";
    
    
  TraceLogger::print_mem("NEST base"); 
  /*if (rank==0) {
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
  
  */
  
  // oberserver variables for validation
  // sum over all after alg has to be equal
  uint64_t n_readSynapses=0;
  uint64_t n_SynapsesInDatasets=0;
  uint64_t n_memSynapses=0;
  
  uint64_t n_conSynapses=0;
  
  tracelogger.begin(0,"run");
  
  CommunicateSynapses_Status com_status=UNSET;
  
  
  H5SynapsesLoader synloader(syn_filename,n_readSynapses,n_SynapsesInDatasets);
  
  
  uint64_t nos = 1e6; 
  
  //load datasets from files
  while (!synloader.eof())
  {
    memPredictor.predictBestLoadNos(nos);
    
    tracelogger.begin(0,"loadSynapses");
    synloader.iterateOverSynapsesFromFiles(synapses_, nos);      
    tracelogger.end(0,"loadSynapses");
      
    tracelogger.begin(0,"sort");
    for (int i=0; i< synapses_.size(); i++)
      synapses_[i].integrateOffset(-1*neuron_id_offset_-1); // inverse NEST offset + csaba offset (offset to 0-indicies)
    
    std::sort(synapses_.begin(), synapses_.end());
    tracelogger.end(0,"sort");
    
    tracelogger.begin(0,"communicate");
    com_status = CommunicateSynapses();
    tracelogger.end(0,"communicate"); 
    
    n_memSynapses+=synapses_.size();
    
    tracelogger.begin(0,"connect");
    threadConnectNeurons(n_conSynapses);
    tracelogger.end(0,"connect");
    
    freeSynapses();
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