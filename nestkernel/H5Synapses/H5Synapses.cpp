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
#include "exceptions.h"

#include "nest_names.h"

#include "poisson_randomdev.h"

#include "communicator.h"

#include <stdio.h>

#ifdef IS_BLUEGENE_Q
#include <spi/include/kernel/memory.h>
#endif

#include <algorithm>


#define _DEBUG_MODE 1


void H5Synapses::singleConnect(NESTNodeSynapse& synapse, nest::index synmodel_id, nest::Node* const target_node, const nest::thread target_thread, uint64_t& n_conSynapses, nestio::Stopwatch::timestamp_t& connect_dur)
{
  nest::index source = synapse.source_neuron_;
  
  // check whether the target is on this process
  if (nest::NestModule::get_network().is_local_node(target_node)) {
    // use region to set allow lock for destroying token
    {
      DictionaryDatum d( new Dictionary );
      
      //new Token is not thread-safe
      //mutex easiest workaround
      omp_set_lock(&tokenLock);
      for (int i=0; i<synapses_.prop_names.size(); i++)
	def< double_t >( d, synapses_.prop_names[i], synapse.prop_values_ [i] );
      omp_unset_lock(&tokenLock);
      
      //set synapse type and check for delay boundary
      
      // current selection of synapse is based on source neuron
      //SynapseModelProperties& synmodel_prop = synmodel_props[0]; //
      
      nestio::Stopwatch::timestamp_t begin= nestio::Stopwatch::get_timestamp();

      bool success = nest::NestModule::get_network().connect(source, target_node->get_gid(), d, synmodel_id);
      
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
    throw nest::IllegalConnection("H5Synapses::singleConnect(): synapse is on wrong node");
  }
}

void H5Synapses::threadConnectNeurons(uint64_t& n_conSynapses)
{
  const int& num_processes = nest::Communicator::get_num_processes();
  const int& num_vp = nest::Communicator::get_num_virtual_processes(); 
  
  uint64_t n_conSynapses_tmp=0;

  std::stringstream ss;
  ss << "threadConnectNeurons new_cons=" << synapses_.size();
  TraceLogger::print_mem(ss.str());
  
  omp_init_lock(&tokenLock);
  
  if (memPredictor.preNESTConnect(synapses_.size())==0)
  {
    
    #pragma omp parallel default(shared) reduction(+:n_conSynapses_tmp)
    {
      nestio::Stopwatch::timestamp_t connect_dur=0;
      nestio::Stopwatch::timestamp_t before_connect=nestio::Stopwatch::get_timestamp();
      
      
      const int tid = nest::NestModule::get_network().get_thread_id();
      
      
      //throw has to be moved out of the parallel region!!
    
      //if (num_vp != (int)(num_processes*omp_get_num_threads()))
      //	throw nest::KernelException ("H5Synapses::threadConnectNeurons(): NEST threads are not equal to OMP threads" );
    
      //without preprocessing:
      //only connect neurons which are on local thread otherwise skip
      
      for (int i=0;i<synapses_.size();i++) {
	
	const nest::index target = synapses_[i].target_neuron_;
	
	//assert for smaller maximum neuron id
	
	nest::Node* const target_node = nest::NestModule::get_network().get_node(target);
	const nest::thread target_thread = target_node->get_thread();
	
	if (target_thread == tid)  // ((synapses_[i].target_neuron_ % num_vp) / num_processes == section_ptr) // synapse belongs to local thread, connect function is thread safe for this condition
	{
	  singleConnect(synapses_[i], synapses_.synmodel_id_, target_node, target_thread, n_conSynapses_tmp, connect_dur);
	}
      }
      tracelogger.store(tid,"nest::connect", before_connect, connect_dur);
    }  
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
      
      
      singleConnect(synapses_[i], synapses_.synmodel_id_, target_node, target_thread, n_conSynapses, connect_dur);
    }
  
    tracelogger.store(0,"nest::connect", before_connect, connect_dur);
  }
  omp_destroy_lock(&tokenLock);
}

/**
 *  Communicate Synpases between the nodes
 *  Aftewards all synapses are on their target nodes
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

/**
 * 
 */
H5Synapses::H5Synapses(nest::index offset, const Name synmodel_name, TokenArray synparam_names): neuron_id_offset_(offset)
{  
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
  synapses_.clear();
}

void H5Synapses::import(const std::string& syn_filename)
{
  int rank = nest::Communicator::get_rank();
  int size = nest::Communicator::get_num_processes();
    
    
  TraceLogger::print_mem("NEST base"); 
  
  // oberserver variables for validation
  // sum over all after alg has to be equal
  uint64_t n_readSynapses=0;
  uint64_t n_SynapsesInDatasets=0;
  uint64_t n_memSynapses=0;
  uint64_t n_conSynapses=0;
  
  tracelogger.begin(0,"run");
  
  CommunicateSynapses_Status com_status=UNSET;
  
  H5SynapsesLoader synloader(syn_filename,n_readSynapses,n_SynapsesInDatasets);
  
  
  //number of synapses per iteration effects memory consumption and speed of the import module
  uint64_t nos = 1e6; 
  
  //load datasets from files
  while (!synloader.eof())
  {
    //number of synapses per iteration might be reduced if there is not enough memory available
    memPredictor.predictBestLoadNos(nos);
    
    tracelogger.begin(0,"loadSynapses");
    synloader.iterateOverSynapsesFromFiles(synapses_, nos);      
    tracelogger.end(0,"loadSynapses");
    
    //integrate offset of ids to synapses
    //
    for (int i=0; i< synapses_.size(); i++)
      synapses_[i].integrateOffset(neuron_id_offset_); // inverse NEST offset + csaba offset (offset to 0-indicies)
    
    tracelogger.begin(0,"sort");
    std::sort(synapses_.begin(), synapses_.end());
    tracelogger.end(0,"sort");
    
    tracelogger.begin(0,"communicate");
    com_status = CommunicateSynapses();
    tracelogger.end(0,"communicate"); 
    
    n_memSynapses+=synapses_.size();
    
    tracelogger.begin(0,"connect");
    threadConnectNeurons(n_conSynapses);
    tracelogger.end(0,"connect");
    
    //freeSynapses();
  }
  
  //recieve datasets from other nodes
  //necessary because datasets may be distributed unbalanced
  while (com_status != NOCOM) {
    tracelogger.begin(0,"communicate");
    com_status = CommunicateSynapses();
    tracelogger.end(0,"communicate");
    
    n_memSynapses+=synapses_.size();
    
    tracelogger.begin(0,"connect");
    threadConnectNeurons(n_conSynapses);
    tracelogger.end(0,"connect");
    
    //freeSynapses();
  }
  
  tracelogger.end(0,"run");
  
  nest::NestModule::get_network().message( SLIInterpreter::M_INFO,
      "H5Synapses::import",
      String::compose( "rank=%1\tn_readSynapses=%2\tn_conSynapses=%3\tn_memSynapses=%4\tn_SynapsesInDatasets=%5",
                          rank, n_readSynapses, n_conSynapses, n_memSynapses, n_SynapsesInDatasets) );
}