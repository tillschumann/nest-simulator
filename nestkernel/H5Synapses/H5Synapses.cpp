#include "H5Synapses.h"

#include <iostream>
//#include "nmpi.h"
#include <algorithm>
#include <dirent.h>
#include <errno.h>
#include <sstream>
#include <sys/types.h>
//#include "timer/stopwatch.h"

#include "compose.hpp"
#include "dictdatum.h"
#include "exceptions.h"
#include "kernel_manager.h"
#include "nest_names.h"
#include "nest_types.h"
#include "nestmodule.h"
#include "node.h"

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

void
H5Synapses::singleConnect( NESTSynapseRef synapse,
  nest::Node* const target_node,
  const nest::thread target_thread,
  DictionaryDatum& d,
  std::vector< const Token* > v_ptr,
  uint64_t& n_conSynapses )
{
  nest::index source = synapse.source_neuron_;

  // safety check whether the target is on this process
  if ( nest::kernel().node_manager.is_local_node( target_node ) )
  {

    // apply kernels to vaues from h5 file
    std::vector< double > values(
      synapse.params_.first(), synapse.params_.end() );
    values = kernel( values );

    assert( values.size() >= 2 );

    const double& delay = values[ 0 ];
    const double& weight = values[ 1 ];
    // fill dictionary with values
    for ( int i = 2; i < values.size(); i++ )
      setValue< double >( *v_ptr[ i ], values[ i ] );

    nest::kernel().connection_manager.connect(
      source, target_node, target_thread, synmodel_id, d, delay, weight );

    n_conSynapses++;
  }
  else
  {
    throw nest::IllegalConnection(
      "H5Synapses::singleConnect(): synapse is on wrong node" );
  }
}

uint64_t
H5Synapses::threadConnectNeurons( uint64_t& n_conSynapses )
{
#ifdef SCOREP_COMPILE
  SCOREP_USER_REGION( "connect", SCOREP_USER_REGION_TYPE_FUNCTION )
#endif

  // get statistics
  uint64_t n_conSynapses_sum = 0;

  std::stringstream ss;
  ss << "threadConnectNeurons\tnew_cons=" << synapses_.size() << "\t"
     << "rank=" << nest::kernel().mpi_manager.get_rank() << "\t"
#if defined IS_BLUEGENE_P || defined IS_BLUEGENE_Q
     << "heap=" << bg_get_heap_mem() << "\t"
     << "stack=" << bg_get_stack_mem() << "\t"
#endif
     << "\n";

  LOG( nest::M_INFO, "H5Synapses::threadConnectNeurons", ss.str() );

#pragma omp parallel default( shared ) reduction( + : n_conSynapses_sum )
  {
    uint64_t n_conSynapses_tmp = 0;
    const int thrd = nest::kernel().vp_manager.get_thread_id();

    // without preprocessing:
    // only connect neurons which are on local thread otherwise skip
    {
      // create DictionaryDatum in region to lock creation and deleting of Token
      // objects
      DictionaryDatum d( new Dictionary );

      // create entries inside of DictionaryDatum and store references to Token
      // objects
      std::vector< const Token* > v_ptr( synapses_.prop_names.size() );
      // not thread safe
      omp_set_lock( &tokenLock );
      for ( int i = 2; i < synapses_.prop_names.size(); i++ )
      {
        def< double >( d, synparam_names[ i ], 0.0 );
        const Token& token_ref = d->lookup2( synparam_names[ i ] );
        v_ptr[ i ] = &token_ref;
      }
      omp_unset_lock( &tokenLock );

      // help variable to only connect subset if stride is larger than 1
      int stride_c = 0;
      for ( int i = 0; i < synapses_.size(); i++ )
      {
        const nest::index target = synapses_[ i ].target_neuron_;
        nest::Node* const target_node =
          nest::kernel().node_manager.get_node( target );
        const nest::thread target_thread = target_node->get_thread();

        // synapse belongs to local thread, connect function is thread safe for
        // this condition
        if ( target_thread == thrd )
        {
          stride_c++;
          // only connect each stride ones
          if ( stride_c == 1 )
            singleConnect( synapses_[ i ],
              target_node,
              target_thread,
              d,
              v_ptr,
              n_conSynapses_tmp );
          if ( stride_c >= stride_ )
            stride_c = 0;
        }
      }
      omp_set_lock( &tokenLock );
    } // lock closing braket to serialize object destroying
    omp_unset_lock( &tokenLock );

    n_conSynapses_sum += n_conSynapses_tmp;
  }
  n_conSynapses += n_conSynapses_sum;
  return n_conSynapses;
}

/**
 *  Communicate Synpases between the nodes
 *  Aftewards all synapses are on their target nodes
 */
CommunicateSynapses_Status
H5Synapses::CommunicateSynapses()
{
#ifdef SCOREP_COMPILE
  SCOREP_USER_REGION( "alltoall", SCOREP_USER_REGION_TYPE_FUNCTION )
#endif
  uint32_t num_processes = nest::kernel().mpi_manager.get_num_processes();

  std::stringstream sstream;
  int sendcounts[ num_processes ], recvcounts[ num_processes ],
    rdispls[ num_processes + 1 ], sdispls[ num_processes + 1 ];
  for ( int32_t i = 0; i < num_processes; i++ )
  {
    sendcounts[ i ] = 0;
    sdispls[ i ] = 0;
    recvcounts[ i ] = -999;
    rdispls[ i ] = -999;
  }

  // use iterator instead
  // uint32_t* send_buffer = new
  // uint32_t[synapses_.size()*synapses_.entry_size_int()];
  std::vector< int > send_buffer;
  send_buffer.reserve( synapses_.size() * synapses_.entry_size_int() );

  // store number of int values per entry
  int entriesadded;
  for ( uint32_t i = 0; i < synapses_.size(); i++ )
  {
    // serialize entry
    entriesadded = synapses_[ i ].serialize( send_buffer );

    // save number of values added
    sendcounts[ synapses_[ i ].node_id_ ] += entriesadded;
  }

  MPI_Alltoall(
    sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD );

  rdispls[ 0 ] = 0;
  sdispls[ 0 ] = 0;
  for ( uint32_t i = 1; i < num_processes + 1; i++ )
  {
    sdispls[ i ] = sdispls[ i - 1 ] + sendcounts[ i - 1 ];
    rdispls[ i ] = rdispls[ i - 1 ] + recvcounts[ i - 1 ];
  }

  // use number of values per entry to determine number of recieved synapses
  const int32_t recv_synpases_count = rdispls[ num_processes ] / entriesadded;

  // allocate recv buffer
  std::vector< int > recvbuf( rdispls[ num_processes ] );

  MPI_Alltoallv( send_buffer.begin(),
    sendcounts,
    sdispls,
    MPI_UNSIGNED,
    recvbuf.begin(),
    recvcounts,
    rdispls,
    MPI_UNSIGNED,
    MPI_COMM_WORLD );

  // fill entries in synapse list
  synapses_.resize( recv_synpases_count );

  // fill synapse list with values from buffer
  for ( uint32_t i = 0; i < synapses_.size(); i++ )
    synapses_[ i ].deserialize( recvbuf );

  // return status
  if ( sdispls[ num_processes ] > 0 && rdispls[ num_processes ] > 0 )
    return SENDRECV;
  else if ( sdispls[ num_processes ] > 0 )
    return SEND;
  else if ( rdispls[ num_processes - 1 ] > 0 )
    return RECV;
  else
    return NOCOM;
}

/**
 *
 */
H5Synapses::H5Synapses( const DictionaryDatum& din )
  : stride_( 1 )
{
  // init lock token
  omp_init_lock( &tokenLock );

  // parse input parameters
  set_status( din );
}

H5Synapses::~H5Synapses()
{
  omp_destroy_lock( &tokenLock );
}

void
H5Synapses::freeSynapses()
{
#ifdef SCOREP_COMPILE
  SCOREP_USER_REGION( "free", SCOREP_USER_REGION_TYPE_FUNCTION )
#endif
  synapses_.clear();
}

void
H5Synapses::integrateMapping()
{
#ifdef SCOREP_COMPILE
  SCOREP_USER_REGION( "det", SCOREP_USER_REGION_TYPE_FUNCTION )
#endif
// integrate mapping from gidcollection
#pragma omp paralalel for
  for ( int i = 0; i < synapses_.size(); i++ )
    synapses_[ i ].integrateMapping( mapping_ );
}
void
H5Synapses::sort()
{
#ifdef SCOREP_COMPILE
  SCOREP_USER_REGION( "sort", SCOREP_USER_REGION_TYPE_FUNCTION )
#endif

  // only needed if there are at least two elements
  if ( synapses_.size() > 1 )
  {
    // arg sort
    typedef std::pair< int, int > intpair;
    std::vector< intpair > v_idx( synapses_.size() );
    for ( int i = 0; i < v_idx.size(); i++ )
    {
      v_idx[ i ].first = synapses_.node_id_[ i ];
      v_idx[ i ].second = i;
    }
    std::sort( rank_idx.begin() rank_idx.end(),
      bool cp(
        const intpair&, const intpair& r ) { return l.first < r.first; } );
    for ( int i = 0; i < rank_idx.size(); i++ )
    {
      rank_idx[ i ].first = i;
    }

    // sort again for fast forward swapping of elements
    std::sort( rank_idx.begin() rank_idx.end(),
      bool cp(
        const intpair&, const intpair& r ) { return l.second < r.second; } );

    // create buf object
    unsigned int source_neuron_tmp;
    unsigned int node_id_tmp;
    std::vector< char > pool_tmp(
      synapses_[ ix ].params_.size() * sizeof( float ) + sizeof( int ) );
    NESTSynapseRef buf( source_neuron_tmp,
      node_id_tmp,
      synapses_[ ix ].params_.size(),
      pool_tmp.begin() );
    // swap elements based on rank ids
    // x -> y
    int last = 0;
    const int ix = rank_idx[ 0 ].first;
    tmp = synapses_[ ix ];

    for ( int i = 1; i < rank_idx.size(); i++ )
    {
      const int ix = rank_idx[ cur ].first;
      synapses_[ ix ].swap( buf );
      last = ix;
    }
  }
}

void
H5Synapses::import( DictionaryDatum& dout )
{
  const int rank = nest::kernel().mpi_manager.get_rank();
  const int size = nest::kernel().mpi_manager.get_num_processes();

  // oberserver variables for validation
  // sum over all after alg has to be equal
  uint64_t n_readSynapses = 0;
  uint64_t n_SynapsesInDatasets = 0;
  uint64_t n_memSynapses = 0;
  uint64_t n_conSynapses = 0;

  CommunicateSynapses_Status com_status = UNSET;

  H5SynapsesLoader synloader( filename_,
    synapses_.prop_names,
    n_readSynapses,
    n_SynapsesInDatasets,
    num_syanpses_per_process_,
    last_total_synapse_ );

  // load datasets from files
  // until end of file
  while ( !synloader.eof() )
  {
    {
#ifdef SCOREP_COMPILE
      SCOREP_USER_REGION_BEGIN( "load", SCOREP_USER_REGION_TYPE_FUNCTION )
#endif
      synloader.iterateOverSynapsesFromFiles( synapses_ );
    }

    integrateMapping();
    sort();
    com_status = CommunicateSynapses();

    // update stats
    n_memSynapses += synapses_.size();

    threadConnectNeurons( n_conSynapses );

    freeSynapses();
  }

  // recieve datasets from other nodes
  // necessary because datasets may be distributed unbalanced
  while ( com_status != NOCOM )
  {
    com_status = CommunicateSynapses();
    n_memSynapses += synapses_.size();
    threadConnectNeurons( n_conSynapses );
    freeSynapses();
  }

  // return values
  def< long >( dout, "readSynapses", n_readSynapses );
  def< long >( dout, "conSynapses", n_conSynapses );
  def< long >( dout, "memSynapses", n_memSynapses );
  def< long >( dout, "SynapsesInDatasets", n_SynapsesInDatasets );
}

void
H5Synapses::set_status( const DictionaryDatum& din )
{
  filename = getValue< std::string >( din, "file" );
  TokenArray isynparam_names = getValue< TokenArray >( din, "params" );
  for ( int i = 0; i < isynparam_names.size(); i++ )
  {
    model_params_.push_back( isynparam_names[ i ] );
  }
  if ( synparam_names.size() < 2 )
    throw BadProperty(
      "parameter list has to contain delay and weight at least" );
  if ( synparam_names[ 0 ] != "delay" )
    throw BadProperty( "first synapse parameter has to be delay" );
  if ( synparam_names[ 1 ] != "weight" )
    throw BadProperty( "second synapse parameter has to be weight" );


  if ( !updateValue< long >(
         din, names::synapses_per_rank, num_syanpses_per_process_ ) )
    num_syanpses_per_process_ = 524288;
  if ( !updateValue< long >( din, names::last_synapse, last_total_synapse_ ) )
    last_total_synapse_ = 0;
  // set stride if set, if not stride is 1
  updateValue< long >( din, "stride", stride_ );

  if ( stride_ < 1 )
    throw BadProperty( "stride has to be one or greater" );

  TokenArray hdf5_names;
  // if set use different names for synapse model and hdf5 dataset columns
  if ( updateValue< TokenArray >( din, names::hdf5_names, hdf5_names ) )
    for ( int i = 0; i < hdf5_names.size(); i++ )
      synapses_.prop_names.push_back( hdf5_names[ i ] );
  else
    synapses_.prop_names = model_params_;

  // use gidcollection as mapping
  // if nothing is set use GIDCollection for all neurons
  // we get an offset of +1
  if ( !updateValue< GIDCollectionDatum >( din, "mapping", mapping_ ) )
  {
    // use all nodes in network
    const int first = 1;
    const int last = nest::kernel().node_manager.size() - 1;
    mapping_ = GIDCollection( first, last );
  }

  // lookup synapse model
  const Name model_name = getValue< Name >( din, "model" );
  const Token synmodel =
    nest::kernel().model_manager.get_synapsedict()->lookup( model_name );
  synmodel_id_ = static_cast< size_t >( synmodel );

  // add kernels
  ArrayDatum kernels;
  if ( updateValue< ArrayDatum >( din, "kernels", kernels ) )
  {
    for ( int i = 0; i < kernels.size(); i++ )
    {
      DictionaryDatum kd = getValue< DictionaryDatum >( kernels[ i ] );
      const std::string kernel_name = getValue< std::string >( kd, "name" );
      const TokenArray kernel_params = getValue< TokenArray >( kd, "params" );
      addKernel( kernel_name, kernel_params );
    }
  }
}


// specified which kernels are available for synapse loading
void
H5Synapses::addKernel( std::string name, TokenArray params )
{
  if ( name == "add" )
    kernel_.push_back< kernel_add< double > >( params );
  else if ( name == "multi" )
    kernel_.push_back< kernel_multi< double > >( params );
  else if ( name == "csaba1" )
    kernel_.push_back< kernel_csaba< double > >( params );
}
