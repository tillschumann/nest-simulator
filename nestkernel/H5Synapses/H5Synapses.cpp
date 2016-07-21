#include "H5Synapses.h"

#include <iostream>      
//#include "nmpi.h"
#include <algorithm> 
#include <sstream>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
//#include "timer/stopwatch.h"

#include "kernel_manager.h"
#include "node.h"
#include "nestmodule.h"
#include "exceptions.h"
#include "compose.hpp"
#include "nest_names.h"

#include "vp_manager_impl.h"

//#include "poisson_randomdev.h"

//#include "communicator.h"

#include <stdio.h>

#ifdef IS_BLUEGENE_Q
#include <spi/include/kernel/memory.h>
#endif

#include <algorithm>

#ifdef SCOREP_USER_ENABLE
#ifndef SCOREP_COMPILE
#define SCOREP_COMPILE
#endif
#include <scorep/SCOREP_User.h>
#endif

#define _DEBUG_MODE 1

void H5Synapses::singleConnect(NESTNodeSynapse& synapse, nest::index synmodel_id, nest::Node* const target_node, const nest::thread target_thread, DictionaryDatum& d ,std::vector<const Token*> v_ptr, uint64_t& n_conSynapses/*, nestio::Stopwatch::timestamp_t& connect_dur*/)
{
  nest::index source = synapse.source_neuron_;
  
  // check whether the target is on this process 
  if (nest::kernel().node_manager.is_local_node(target_node)) {
    //nestio::Stopwatch::timestamp_t begin= nestio::Stopwatch::get_timestamp(); 

    std::vector<double> values(synapse.prop_values_, synapse.prop_values_+5);
    values = kernel(values);
    
    for (int i=2; i<values.size(); i++) {
      setValue<double_t>( *v_ptr[i], values[i] );
    }

    nest::kernel().connection_manager.connect(source, target_node, target_thread, synmodel_id, d, values[0], values[1]);
    
    n_conSynapses++;
  }
  else
  {
    throw nest::IllegalConnection("H5Synapses::singleConnect(): synapse is on wrong node");
  }
}

uint64_t H5Synapses::threadConnectNeurons(uint64_t& n_conSynapses)
{
#ifdef SCOREP_COMPILE 
  SCOREP_USER_REGION("connect", SCOREP_USER_REGION_TYPE_FUNCTION)
#endif
  const int rank = nest::kernel().mpi_manager.get_rank();
  const int num_processes = nest::kernel().mpi_manager.get_num_processes();
  const int num_vp = nest::kernel().vp_manager.get_num_virtual_processes();
  uint64_t n_conSynapses_sum=0;
  uint64_t n_conSynapses_max=0;

 
  
  uint64_t shared, persist, heapavail, stackavail, stack, heap, guard, mmap;
    Kernel_GetMemorySize(KERNEL_MEMSIZE_SHARED, &shared);
    Kernel_GetMemorySize(KERNEL_MEMSIZE_PERSIST, &persist);
    Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAPAVAIL, &heapavail);
    Kernel_GetMemorySize(KERNEL_MEMSIZE_STACKAVAIL, &stackavail);
    Kernel_GetMemorySize(KERNEL_MEMSIZE_STACK, &stack);
    Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAP, &heap);
    Kernel_GetMemorySize(KERNEL_MEMSIZE_GUARD, &guard);
    Kernel_GetMemorySize(KERNEL_MEMSIZE_MMAP, &mmap);
    
    std::stringstream ss;
    ss << "threadConnectNeurons\tnew_cons=" << synapses_.size() << "\t"
       << "rank=" << nest::kernel().mpi_manager.get_rank() << "\t"
       << "heap=" << static_cast<double>(heap)/(1024*1024) << "\t"
       << "stack=" << static_cast<double>(stack)/(1024*1024) << "\t"
       << "havail=" << static_cast<double>(heapavail)/(1024*1024) << "\t"
       << "savail=" << static_cast<double>(stackavail)/(1024*1024) << "\t"
       << "shared=" << static_cast<double>(shared)/(1024*1024) << "\t"
       << "persist=" << static_cast<double>(persist)/(1024*1024) << "\t"
       << "guard=" << static_cast<double>(guard)/(1024*1024) << "\t"
       << "mmap=" << static_cast<double>(mmap)/(1024*1024) << "\n";

    LOG( nest::M_DEBUG, "H5Synapses::threadConnectNeurons", ss.str());

  #pragma omp parallel default(shared) reduction(+:n_conSynapses_sum) reduction(max:n_conSynapses_max)
  {
    uint64_t n_conSynapses_tmp = 0;
    //nestio::Stopwatch::timestamp_t connect_dur=0;
    //nestio::Stopwatch::timestamp_t before_connect=nestio::Stopwatch::get_timestamp();
    const int tid = nest::kernel().vp_manager.get_thread_id();

    /*if (num_vp != (int)(num_processes*omp_get_num_threads())) {
    	std::stringstream ss;
		ss << "Thread number is wrong\tnest-threads=" << num_vp << "\t"
		   << "openmp-threads=" << num_processes*omp_get_num_threads() << "\n";
		LOG(nest::M_ERROR, "H5Synapses::threadConnectNeurons", ss.str() );
    }*/
  
    //without preprocessing:
    //only connect neurons which are on local thread otherwise skip
    {
      // create DictionaryDatum in region to lock creation and deleting of Token objects
      DictionaryDatum d( new Dictionary );
      
      //create entries inside of DictionaryDatum and store references to Token objects
      std::vector<const Token*> v_ptr(synapses_.prop_names.size());
      omp_set_lock(&tokenLock);
      for (int i=2; i<synapses_.prop_names.size(); i++) {
		def< double_t >( d, synparam_names[i], param_offset[i]  );
		const Token& token_ref = d->lookup2( synparam_names[i] );
		v_ptr[i] = &token_ref;
      }
      omp_unset_lock(&tokenLock);
	
    int stride_c=0;

    for (int i=0;i<synapses_.size();i++) {
	const nest::index target = synapses_[i].target_neuron_;
	try
	{
	  nest::Node* const target_node = nest::kernel().node_manager.get_node(target);
	  const nest::thread target_thread = target_node->get_thread();
	  
	  if (target_thread == tid)  // ((synapses_[i].target_neuron_ % num_vp) / num_processes == section_ptr) // synapse belongs to local thread, connect function is thread safe for this condition
	  {
		stride_c++;
		if (stride_c==1) {
			singleConnect(synapses_[i], synapses_.synmodel_id_, target_node, target_thread, d, v_ptr, n_conSynapses_tmp/*, connect_dur*/);
		}
		if (stride_c>=stride_) {
			stride_c = 0;
	    }
	  }
	}
	catch (nest::UnknownNode e)
	{
	  LOG( nest::M_ERROR,
	    "H5Synapses::threadConnectNeurons",
	    String::compose( "UnknownNode\trank=%1\t%2",
				rank, e.message()) );
	}
	catch (nest::IllegalConnection e)
	{
		LOG( nest::M_ERROR,
	    "H5Synapses::threadConnectNeurons",
	    String::compose( "IllegalConnection\trank=%1\t%2",
				rank, e.message()) );
	}
	catch (nest::KernelException e)
	{
		LOG( nest::M_ERROR,
	    "H5Synapses::threadConnectNeurons",
	    String::compose( "KernelException\trank=%1\t%2",
				rank, e.message()) );
	}
      }
      //tracelogger.store(tid,"nest::connect", before_connect, connect_dur);
      omp_set_lock(&tokenLock);
    }  // lock closing braket to serialize object destroying
    omp_unset_lock(&tokenLock);

    n_conSynapses_sum += n_conSynapses_tmp;
    n_conSynapses_max = n_conSynapses_tmp;
  }
  n_conSynapses += n_conSynapses_sum;
  return n_conSynapses;
}

void H5Synapses::ConnectNeurons(uint64_t& n_conSynapses)
{
  int num_processes = nest::kernel().mpi_manager.get_num_processes();
  
  //nestio::Stopwatch::timestamp_t connect_dur=0;
  //nestio::Stopwatch::timestamp_t before_connect=nestio::Stopwatch::get_timestamp();
  
  //omp_init_lock(&tokenLock);
  
  {
    DictionaryDatum d( new Dictionary );
	
	//new Token is not thread-safe
    //mutex easiest workaround
    omp_set_lock(&tokenLock);
    for (int i=2; i<synapses_.prop_names.size(); i++)
      def< double_t >( d, synparam_names[i], param_offset[i]  );
    omp_unset_lock(&tokenLock);
    
    //if (memPredictor.preNESTConnect(synapses_.size())==0)
    //{
      for (int i=0; i< synapses_.size(); i++) {
	const nest::index target = synapses_[i].target_neuron_;
	nest::Node* const target_node = nest::kernel().node_manager.get_node(target);
	const nest::thread target_thread = target_node->get_thread();
	
	
	//singleConnect(synapses_[i], synapses_.synmodel_id_, target_node, target_thread, d, , n_conSynapses, connect_dur);
      }
      omp_set_lock(&tokenLock);
  }
  omp_unset_lock(&tokenLock);
  
  //tracelogger.store(0,"nest::connect", before_connect, connect_dur);
  //}
  //omp_destroy_lock(&tokenLock);
}

/**
 *  Communicate Synpases between the nodes
 *  Aftewards all synapses are on their target nodes
 */
CommunicateSynapses_Status H5Synapses::CommunicateSynapses()
{
#ifdef SCOREP_COMPILE
  SCOREP_USER_REGION("alltoall", SCOREP_USER_REGION_TYPE_FUNCTION)
#endif
  uint32_t num_processes = nest::kernel().mpi_manager.get_num_processes();
  
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
  
  //tracelogger.begin(0, "mpi wait"); 
  MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
  //tracelogger.end(0, "mpi wait");
  
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
H5Synapses::H5Synapses(const Name synmodel_name, TokenArray isynparam_names)
: neuron_id_offset_(1), stride_(1)
{  
  //init lock token
  omp_init_lock(&tokenLock);
  
  for (int i=0; i<isynparam_names.size(); i++) {
    synparam_names.push_back(isynparam_names[i]);
  }

  //lookup synapse model
  const Token synmodel = nest::kernel().model_manager.get_synapsedict()->lookup(synmodel_name);
  synapses_.synmodel_id_ = static_cast<nest::index>(synmodel);
}

H5Synapses::~H5Synapses()
{
  omp_destroy_lock(&tokenLock);
}

void H5Synapses::freeSynapses()
{
#ifdef SCOREP_COMPILE
  SCOREP_USER_REGION("free", SCOREP_USER_REGION_TYPE_FUNCTION)
#endif
  synapses_.clear();
}

void H5Synapses::import(const std::string& syn_filename, const DictionaryDatum& d)
{
  uint64_t num_syanpses_per_process = 0;
  uint64_t last_total_synapse = 0;
  TokenArray hdf5_names;

  updateValue< uint64_t >( d, names::synapses_per_rank, num_syanpses_per_process );
  updateValue< uint64_t >( d, names::last_synapse, last_total_synapse );

  //if set use different names for synapse model and hdf5 dataset columns
  if (updateValue< TokenArray >( d, names::hdf5_names, hdf5_names)) {
	  for (int i=0; i<hdf5_names.size(); i++) {
	      synapses_.prop_names.push_back(hdf5_names[i]);
	    }
  }
  else {
	  synapses_.prop_names = synparam_names;
  }

  int rank = nest::kernel().mpi_manager.get_rank();
  int size = nest::kernel().mpi_manager.get_num_processes();
  
  // oberserver variables for validation
  // sum over all after alg has to be equal
  uint64_t n_readSynapses=0;
  uint64_t n_SynapsesInDatasets=0;
  uint64_t n_memSynapses=0;
  uint64_t n_conSynapses=0;
  
  CommunicateSynapses_Status com_status=UNSET;
  
  H5SynapsesLoader synloader(syn_filename, synapses_.prop_names,n_readSynapses,n_SynapsesInDatasets, num_syanpses_per_process, last_total_synapse);
  //number of synapses per iteration effects memory consumption and speed of the import module
  //uint64_t nos = 1e6; 
  
  //load datasets from files
  while (!synloader.eof())
  {
    {
#ifdef SCOREP_COMPILE
      SCOREP_USER_REGION("load", SCOREP_USER_REGION_TYPE_FUNCTION)
#endif
      synloader.iterateOverSynapsesFromFiles(synapses_);   
    }

    {
#ifdef SCOREP_COMPILE
      SCOREP_USER_REGION("det", SCOREP_USER_REGION_TYPE_FUNCTION)
#endif
    for (int i=0; i< synapses_.size(); i++)
      synapses_[i].integrateOffset(neuron_id_offset_); // inverse NEST offset + csaba offset (offset to 0-indicies)
    }
    
    {
#ifdef SCOREP_COMPILE
      SCOREP_USER_REGION("sort", SCOREP_USER_REGION_TYPE_FUNCTION)
#endif
    std::sort(synapses_.begin(), synapses_.end());
    }

    com_status = CommunicateSynapses();
    
    n_memSynapses+=synapses_.size();

    uint64_t n_new_con = threadConnectNeurons(n_conSynapses);
    
    freeSynapses();
  }
  
  //recieve datasets from other nodes
  //necessary because datasets may be distributed unbalanced
  while (com_status != NOCOM) {
    com_status = CommunicateSynapses();
    n_memSynapses+=synapses_.size();
    threadConnectNeurons(n_conSynapses);
    freeSynapses();
  }
  
  //tracelogger.end(0,"run");
  
  LOG (nest::M_INFO,
      "H5Synapses::import",
      String::compose( "rank=%1\tn_readSynapses=%2\tn_conSynapses=%3\tn_memSynapses=%4\tn_SynapsesInDatasets=%5",
                          rank, n_readSynapses, n_conSynapses, n_memSynapses, n_SynapsesInDatasets) );
}

void H5Synapses::set_status( const DictionaryDatum& d ) {
	//use gid collection as mapping
	if (!updateValue< GIDCollectionDatum >( d, names::neurons, neurons )) {
        const nest::index last_neuron = nest::kernel().node_manager.size();
		neurons = GIDCollectionDatum(1, last_neuron);
	}
    //set stride if set, if not stride is 1
    updateValue<size_t>(d, nest::stride, stride_);
    
    //add kernels
    ArrayDatum kernels;
    if (updateValue<ArrayDatum>(d, nest::kernels, kernels)) {
        for (int i=0; i< kernels.size(); i++) {
            DictionaryDatum kd = getValue< DictionaryDatum >( kernels[i] );
            const std::string kernel_name = getValue<std::string>( kd, nest::name );
            const std::string kernel_params = getValue<std::string>( kd, nest::name );
            addKernel(kernel_name, kernel_params);
        }
    }
}

void H5Synapses::addKernel(std::string name, TokenArray params)
{
	if (name == "add") {
		std::vector<float> v(params.size());
		for (int i=0; i<params.size(); i++)
			v[i] = params[i];
		kernel.push_back< kernel_add<float> >(v);
	}
	else if (name == "multi") {
		std::vector<float> v(params.size());
		for (int i=0; i<params.size(); i++)
			v[i] = params[i];
		kernel.push_back< kernel_multi<float> >(v);
	}
	else if (name == "csaba1") {
		std::vector<float> v(params.size());
		for (int i=0; i<params.size(); i++)
			v[i] = params[i];
		kernel.push_back< kernel_csaba<float> >(v);
	}

}
