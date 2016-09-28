#include "H5Synapses/H5Synapses.h"
#include "H5Synapses/SynapseList.h"

#include "kernel_manager.h"
#include "node.h"
#include "nestmodule.h"
#include "exceptions.h"
#include "compose.hpp"
#include "nest_names.h"
#include "nest_types.h"
#include "dictdatum.h"
#include "vp_manager_impl.h"

#include <iostream>
#include <algorithm> 
#include <sstream>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <ctime>
#include <sys/time.h>
#include <stdio.h>
#include <queue>
#include <algorithm>

#ifdef IS_BLUEGENE_Q
#include <spi/include/kernel/memory.h>
#endif

#define _DEBUG_MODE 1

using namespace h5import;

void H5Synapses::singleConnect( const SynapseRef& synapse, nest::index synmodel_id, nest::Node* target_node, nest::thread target_thread, DictionaryDatum& d ,std::vector<const Token*> v_ptr, uint64_t& n_conSynapses/*, nestio::Stopwatch::timestamp_t& connect_dur*/)
{
  nest::index source = synapse.source_neuron_;
  
  // safety check whether the target is on this process
  //if (nest::kernel().node_manager.is_local_node(target_node)) {
    //std::vector<double>& values = *(kernel_( synapse.params_.begin(), synapse.params_.end() ));

    //const double& delay = (values)[0];
    //const double& weight = (values)[1];
    //for (int i=2; i<values.size(); i++)
    //  setValue<double>( *v_ptr[i], (values)[i] );

    const double delay = *(synapse.params_.begin());
    const double weight = *(synapse.params_.begin()+1);

    for (int i=2; i<model_params_.size(); i++)
       setValue<double>( *v_ptr[i], *(synapse.params_.begin()+i) );

    nest::kernel().connection_manager.connect(source, target_node, target_thread, synmodel_id, d, delay, weight);

    n_conSynapses++;
  //}
  //else
  //{
  //  throw nest::IllegalConnection("H5Synapses::singleConnect(): synapse is on wrong node");
  //}
}

void H5Synapses::threadConnectNeurons(SynapseList& synapses, uint64_t& n_conSynapses)
{
	const int rank = nest::kernel().mpi_manager.get_rank();
	const int num_processes = nest::kernel().mpi_manager.get_num_processes();
	const int num_vp = nest::kernel().vp_manager.get_num_virtual_processes();
	uint64_t n_conSynapses_sum=0;

	#pragma omp parallel default(shared) reduction(+:n_conSynapses_sum)
	{
		uint64_t n_conSynapses_tmp=0;
		const int tid = nest::kernel().vp_manager.get_thread_id();

		{
			// create DictionaryDatum in region to lock creation and deleting of Token objects
			DictionaryDatum d( new Dictionary );

			//create entries inside of DictionaryDatum and store references to Token objects
			std::vector<const Token*> v_ptr(model_params_.size());
			omp_set_lock(&tokenLock_);
			for (int i=2; i<model_params_.size(); i++) {
				def< double >( d, model_params_[i], 0.0  );
				const Token& token_ref = d->lookup2( model_params_[i] );
				v_ptr[i] = &token_ref;
			}
			omp_unset_lock(&tokenLock_);

			//only connect neurons for stride 0
			int stride_c=0;
			for (int i=0;i<synapses.size();i++) {
				const SynapseRef syn = synapses[i];

				const nest::index target = syn.target_neuron_;
				//try
				//{
					nest::Node* const target_node = nest::kernel().node_manager.get_node(target);
					const nest::thread target_thread = target_node->get_thread();

					//only connect neurons which are on local thread otherwise skip
					if (target_thread == tid)
					{
						//stride_c++;
						//if (stride_c==1)
						singleConnect( syn, synmodel_id_, target_node, target_thread, d, v_ptr, n_conSynapses_tmp );
						//if (stride_c>=stride_)
						//stride_c = 0;
					}
				//}
				/*catch ( nest::KernelException& e ) {
					std::cout << "KernelException rank=" << rank << " source=" << syn.source_neuron_ <<" target=" << target << " message: " << e.message() << std::endl;
				}
				catch ( std::exception& ex ) {
					std::cout << "exception rank=" << rank << " source=" << syn.source_neuron_ <<" target=" << target << " what: " << ex.what() << '\n';
				}
				catch ( ... ) {
					std::cout << "ERROR" << std::endl;
				}*/
			}
			omp_set_lock(&tokenLock_);
		}  // lock closing braket to serialize object destroying
		omp_unset_lock(&tokenLock_);
	
		n_conSynapses_sum += n_conSynapses_tmp;
	}
	n_conSynapses += n_conSynapses_sum;
}

/**
 *  Communicate Synpases between the nodes
 *  Aftewards all synapses are on their target nodes
 */
CommunicateSynapses_Status
H5Synapses::CommunicateSynapses( SynapseList& synapses )
{
	uint32_t num_processes = kernel().mpi_manager.get_num_processes();

	int sendcounts[ num_processes ], recvcounts[ num_processes ],
	rdispls[ num_processes + 1 ], sdispls[ num_processes + 1 ];
	for ( int32_t i = 0; i < num_processes; i++ )
	{
		sendcounts[ i ] = 0;
		sdispls[ i ] = 0;
		recvcounts[ i ] = -999;
		rdispls[ i ] = -999;
	}

	const int intsizeof_entry = synapses.sizeof_entry()/sizeof(int);
	mpi_buffer<int> send_buffer(synapses.size() * intsizeof_entry, true);

	// store number of int values per entry
	int entriesadded;

	#pragma omp parallel for
	for ( size_t i = 0; i < synapses.size(); i++ )
	{
		const size_t offset = i * intsizeof_entry;
		// serialize entry
		entriesadded = synapses[ i ].serialize( send_buffer, offset );

		// save number of values added
		#pragma omp atomic
		sendcounts[ synapses[ i ].node_id_ ] += entriesadded;
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
	const int32_t recv_synpases_count = rdispls[ num_processes ] / intsizeof_entry;

	// allocate recv buffer
	mpi_buffer<int> recvbuf( rdispls[ num_processes ], true );

	MPI_Alltoallv( send_buffer.begin(),
	sendcounts,
	sdispls,
	MPI_INT,
	recvbuf.begin(),
	recvcounts,
	rdispls,
	MPI_INT,
	MPI_COMM_WORLD );

	// make sure that enough entries are allocaed in list
	synapses.resize( recv_synpases_count );

	// fill synapse list with values from buffer
	#pragma omp parallel for
	for ( size_t i = 0; i < synapses.size(); i++ ) {
		const size_t offset = i * intsizeof_entry;
		synapses[ i ].deserialize( recvbuf, offset );
	}
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

void H5Synapses::integrateMapping( SynapseList& synapses )
{
	for ( int i = 0; i < synapses.size(); i++ ) {
		SynapseRef s = synapses[i];
		s.source_neuron_ = mapping_[ s.source_neuron_ ];
		s.target_neuron_ = mapping_[ s.target_neuron_ ];

		//mapping based on the target neuron id
		const thread vp = kernel().vp_manager.suggest_vp( s.target_neuron_ );
		s.node_id_ = kernel().mpi_manager.get_process_id( vp );
	}
}

//helper for sort
typedef std::pair< int, int > intpair;
inline bool first_less( const intpair& l, const intpair& r ) { return l.first < r.first; };

void
H5Synapses::sort( SynapseList& synapses )
{
    if ( synapses.size() > 1 )
    {
        // arg sort
        std::vector< intpair > v_idx( synapses.size() );
        for ( int i = 0; i < v_idx.size(); i++ )
        {
            v_idx[ i ].first = synapses[ i ].node_id_;
            v_idx[ i ].second = i;
        }
        std::sort( v_idx.begin(), v_idx.end(), first_less);

        // create buf object
        uint32_t source_neuron_tmp;
        uint32_t node_id_tmp;
        std::vector< char > pool_tmp( synapses.sizeof_entry() );
        SynapseRef buf( source_neuron_tmp,
                node_id_tmp,
                synapses.get_num_params(),
                &pool_tmp[0] );

        //apply reordering based on v_idx[:].second
        size_t i, j, k;
        for(i = 0; i < synapses.size(); i++){
            if(i != v_idx[ i ].second){
                buf = synapses[i];
                k = i;
                while(i != (j = v_idx[ k ].second)){
                    synapses[k] = synapses[j];
                    v_idx[ k ].second = k;
                    k = j;
                }
                synapses[k] = buf;
                v_idx[ k ].second = k;
            }
        }
    }
}

/**
 * 
 */
H5Synapses::H5Synapses(const DictionaryDatum& din)
: stride_(1)//, kernel_(nest::kernel().vp_manager.get_num_threads())
{  
	//init lock token
	omp_init_lock(&tokenLock_);

	//parse input parameters
	set_status(din);
}

H5Synapses::~H5Synapses()
{
	omp_destroy_lock(&tokenLock_);
}

void H5Synapses::import(DictionaryDatum& dout)
{
	int rank = nest::kernel().mpi_manager.get_rank();
	int size = nest::kernel().mpi_manager.get_num_processes();
  
	// oberserver variables for validation
	// sum over all after alg has to be equal
	uint64_t n_readSynapses=0, n_memSynapses=0, n_conSynapses=0, n_SynapsesInDatasets=0;
	struct timeval start_mpicon, end_mpicon, start_load, end_load, start_push, end_push, start_con, end_con;
	long long t_load=0, t_mpicon=0, t_push=0, t_con=0;
  
	CommunicateSynapses_Status com_status=UNSET;
  
	//open hdf5 file
	h5reader synloader( filename_, h5comp_params_, transfersize_, sizelimit_ );
	n_SynapsesInDatasets += synloader.size();

	std::queue< SynapseList* > synapse_queue;

     // add all synapses into queue
     gettimeofday(&start_push, NULL);
	#pragma omp parallel
    {
		#pragma omp single
    	 {
    		 while ( !synloader.eof() ) {
				 SynapseList* newone = new SynapseList( h5comp_params_.size() );
        
    			 h5reader::h5view dataspace_view;

    			 gettimeofday(&start_load, NULL);
    			 synloader.readblock( *newone, dataspace_view );
    			 gettimeofday(&end_load, NULL);

    			 n_readSynapses += newone->size();
    			 t_load += (1000 * (end_load.tv_sec - start_load.tv_sec))
                		+ ((end_load.tv_usec - start_load.tv_usec) / 1000);

    			 //spawn task
         	 	 #pragma omp task firstprivate(newone, dataspace_view)
    			 {
    				 synloader.integrateSourceNeurons( *newone, dataspace_view );
    				 integrateMapping(*newone);
    				 sort(*newone);
    			 }
    			 synapse_queue.push(newone);
    		 }
    	 }
     	 #pragma omp taskwait
	 }
     gettimeofday(&end_push, NULL);

     //iterate over queue and connect connections in NEST data structure
	while (!synapse_queue.empty()) {
		gettimeofday(&start_mpicon, NULL);
		SynapseList* synapses = synapse_queue.front();
		synapse_queue.pop();

		com_status = CommunicateSynapses(*synapses);

		// update stats after communication
		n_memSynapses += synapses->size();

		gettimeofday(&start_con, NULL);
		threadConnectNeurons( *synapses, n_conSynapses );
		gettimeofday(&end_con, NULL);

		delete synapses;
		gettimeofday(&end_mpicon, NULL);

		t_con += (1000 * (end_con.tv_sec - start_con.tv_sec))
                   + ((end_con.tv_usec - start_con.tv_usec) / 1000);

		t_mpicon += (1000 * (end_mpicon.tv_sec - start_mpicon.tv_sec))
		   + ((end_mpicon.tv_usec - start_mpicon.tv_usec) / 1000);
	}

	t_push += (1000 * (end_push.tv_sec - start_push.tv_sec))
		   + ((end_push.tv_usec - start_push.tv_usec) / 1000);

    std::cout << "rank=" << nest::kernel().mpi_manager.get_rank()
   			  << "\tread_cons="<< n_readSynapses
   			  << "\tmem_cons=" << n_memSynapses
   			  << "\tcon_cons=" << n_conSynapses
   			  << "\tt_load=" << t_load  << "ms"
   			  << "\tt_mpicon=" << t_mpicon << "ms"
   			  << "\tt_push=" << t_push << "ms"
			  << "\tt_con=" << t_con << "ms"
           	  << std::endl;
  

    def< long >( dout, "readSynapses",  n_readSynapses);
	def< long >( dout, "conSynapses",  n_conSynapses);
	def< long >( dout, "memSynapses",  n_memSynapses);
	def< long >( dout, "SynapsesInDatasets", n_SynapsesInDatasets);
}

void H5Synapses::set_status( const DictionaryDatum& din ) {
	filename_ = getValue< std::string >(din, "file");
	TokenArray isynparam_names = getValue<TokenArray>(din, "params");
	for (int i=0; i<isynparam_names.size(); i++) {
	  model_params_.push_back(isynparam_names[i]);
	}
	if (model_params_.size()<2)
		throw BadProperty("parameter list has to contain delay and weight at least");
	if (model_params_[0] != "delay")
		throw BadProperty("first synapse parameter has to be delay");
	if (model_params_[1] != "weight")
		throw BadProperty("second synapse parameter has to be weight");


	if (!updateValue< long >( din, names::synapses_per_rank, transfersize_ ))
		transfersize_ = 524288;
	if (!updateValue< long >( din, names::last_synapse, sizelimit_ ))
		sizelimit_ = -1;
	//set stride if set, if not stride is 1
        if (!updateValue<long>(din, "stride", stride_))
		stride_ = 1;

	if (stride_<1)
		throw BadProperty("stride has to be one or greater");

	TokenArray hdf5_names;
	//if set use different names for synapse model and hdf5 dataset columns
	if (updateValue< TokenArray >( din, names::hdf5_names, hdf5_names)) {
	  for (int i=0; i<hdf5_names.size(); i++) {
		  h5comp_params_.push_back(hdf5_names[i]);
		}
	}
	else {
		h5comp_params_ = model_params_;
	}
	//if nothing is set use GIDCollection for all neurons
	//we get an offset of +1
	if (!updateValue< GIDCollectionDatum >( din, "mapping", mapping_)) {
		//use all nodes in network
		const int first = 1;
		const int last = nest::kernel().node_manager.size()-1;
		mapping_ = GIDCollection( first,last );
	}

	//lookup synapse model
	const Name model_name = getValue<Name>(din, "model");
	const Token synmodel = nest::kernel().model_manager.get_synapsedict()->lookup(model_name);
	synmodel_id_ = static_cast<size_t>(synmodel);
    
    //add kernels
    ArrayDatum kernels;
    if (updateValue<ArrayDatum>(din, "kernels", kernels)) {
        for (int i=0; i< kernels.size(); i++) {
            DictionaryDatum kd = getValue< DictionaryDatum >( kernels[i] );
            const std::string kernel_name = getValue<std::string>( kd, "name" );
            const TokenArray kernel_params = getValue<TokenArray>( kd, "params" );
            addKernel(kernel_name, kernel_params);
        }
    }
}

void H5Synapses::addKernel( std::string name, TokenArray params )
{
	if (name == "add" )
		kernel_.push_back< kernel_add< double > >( params );
	else if ( name == "multi" )
		kernel_.push_back< kernel_multi< double > >( params );
	else if ( name == "srwa" )
		kernel_.push_back< kernel_srwa< double > >( params );
}
