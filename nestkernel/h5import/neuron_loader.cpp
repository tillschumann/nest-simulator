#include <string>

#include "node.h"
#include "nestmodule.h"
#include "nest_names.h"
#include "kernel_manager.h"

#include "h5import/neuron_loader.h"
#include "h5import/h5neuronfile.h"


NeuronLoader::NeuronLoader(const DictionaryDatum& din)
{
    filename_ = getValue< std::string >(din, "file");
    TokenArray param_names = getValue<TokenArray>(din, "params");
    for ( size_t i=0; i<param_names.size(); i++)
        model_param_names_.push_back(param_names[i]);

    //if params from file set use different parameters
    TokenArray toh5params;
    if ( updateValue<TokenArray>( din, "params_read_from_file", toh5params ) ) {
    	std::vector< std::string > h5params;
    	for ( size_t i=0; i<toh5params.size(); i++ )
			 h5params.push_back( toh5params[ i ] );
    	neurons_.setParameters( h5params );
    }
    else
    	neurons_.setParameters( model_param_names_ );

    const Name model_name = getValue<Name>(din, "model");
    const Token neuron_model = nest::kernel().model_manager.get_modeldict()->lookup(model_name);
    neurons_.model_id_ = static_cast< nest::index >(neuron_model);
    
    std::string subnet_name = "";
    if (updateValue<std::string>(din, "subnet", subnet_name)) {
        neurons_.with_subnet = (subnet_name != "");
        if (neurons_.with_subnet)
            neurons_.subnet_name = subnet_name;
    }
    else
    	neurons_.with_subnet = false;

    //add kernels
    ArrayDatum kernels;
    if (updateValue<ArrayDatum>(din, "kernels", kernels)) {
        for ( size_t i=0; i< kernels.size(); i++) {
            DictionaryDatum kd = getValue< DictionaryDatum >( kernels[i] );
            const std::string kernel_name = getValue< std::string >( kd, "name" );
            const TokenArray kernel_params = getValue< TokenArray >( kd, "params" );
            addKernel(kernel_name, kernel_params);
        }
    }
}

void NeuronLoader::addKernel(const std::string& name, TokenArray params)
{
	if (name == "add")
		kernel_.push_back< kernel_add<float> >(params);
	if (name == "multi")
		kernel_.push_back< kernel_multi<float> >(params);
}

void NeuronLoader::execute(DictionaryDatum& dout)
{
    H5NeuronFile cellLoader( filename_ );

    const  uint64_t non= cellLoader.getNumberOfCells(neurons_.parameter_names[0]);
    neurons_.resize(non);

    //the id of the fist new created neuron
    nest::index first_neuron = nest::kernel().node_manager.size();

    //loads all parameters for the local neurons based on NEST neuron distribution
    cellLoader.loadLocalParameters(neurons_.parameter_names, non, first_neuron, neurons_);
    GIDCollectionDatum added_neurons = CreateNeurons();
    def< GIDCollectionDatum >( dout, "added_gids", added_neurons );

    if (neurons_.with_subnet) {
        cellLoader.loadSubnets(non, neurons_.subnet_name, neurons_);
        def< GIDCollectionDatum >( dout, "subnet", CreateSubnets(added_neurons) );
    }
}

/*
 * Create subnets for each unique entry inside 'subnet' dataset
 */
GIDCollectionDatum NeuronLoader::CreateSubnets(const GIDCollectionDatum& added_neurons)
{
    //find all subnets
	std::vector< std::vector<long> > gids;
    std::vector< int > unique_subnets;

    for ( size_t i=0; i<neurons_.size(); i++) {
    	if (neurons_[i].subnet_ != 0) {
    		size_t index = std::distance(unique_subnets.begin(), std::find(unique_subnets.begin(), unique_subnets.end(), neurons_[i].subnet_));
			if (index==unique_subnets.size()) {
				unique_subnets.push_back(neurons_[i].subnet_);
				gids.push_back(std::vector<long>());
				index = unique_subnets.size()-1;
			}
			gids[index].push_back(added_neurons[i]);
    	}
    }


    //only supports one collection so far
    return GIDCollection(TokenArray(gids[0]));
}

/*
 * Create Neurons in subnets using loaded parameters
 */
GIDCollectionDatum NeuronLoader::CreateNeurons()
{
    const size_t non = neurons_.size();
    const nest::index first_neuron_gid = nest::kernel().node_manager.size();
    const nest::index last_neuron_gid = nest::kernel().node_manager.add_node(neurons_.model_id_, non);
    
    GIDCollectionDatum added_neurons = GIDCollection(first_neuron_gid, last_neuron_gid);
    
    //set parameters of created neurons
    //ids of neurons are continously even though they might be in different subnets
    for ( size_t i=0; i<non; i++ ) {
    	const int gid = added_neurons[i];
        if (nest::kernel().node_manager.is_local_gid(gid)) {
			nest::Node* node = nest::kernel().node_manager.get_node(gid);
			if (node->is_local()) {
				DictionaryDatum d( new Dictionary );
				NeuronObj Nnn = neurons_[i];
				std::vector<float>* values = kernel_( Nnn.params_.begin(), Nnn.params_.end() );

				//copy values into sli data objects
				for ( size_t j=0; j<model_param_names_.size(); j++) {
				   def< double >( d, model_param_names_[j], (*values)[j] );
				}
				//pass sli objects to neuron
				node->set_status(d);
			}
        }
    }
    return added_neurons;
}
