#include <string>

#include "H5Neurons.h"

#include "node.h"
#include "nestmodule.h"
#include "nest_names.h"
#include "kernel_manager.h"

#include "H5CellLoader.h"



H5Neurons::H5Neurons(const DictionaryDatum& din)
{
    filename = getValue<TokenArray>(din, names::file);
    TokenArray param_names = getValue<TokenArray>(din, names::params);
    for (int i=0; i<param_names.size(); i++) {
        model_param_names.push_back(param_names[i]);
    }
    TokenArray h5params = param_names;
    
    //if params from file set use different parameters
    updateValue<TokenArray>(din, names::params_from_file, h5params);
    for (int i=0; i<h5params.size(); i++) {
        neurons_.parameter_names.push_back(h5params[i]);
    }
    
    const Name model_name = getValue<TokenArray>(din, names::model);
    const Token neuron_model = kernel().model_manager.get_modeldict()->lookup(model_name);
    neurons_.model_id_ = static_cast< index >(model_name);
    
    std::string subnet_name = ""
    if (updateValue<std::string>(din, names::subnet, subnet_name)) {
        neurons_.with_subnet = (subnet_name != "");
        if (neurons_.with_subnet)
            neurons_.subnet_name = subnet_name.toString();
    }
    //add kernels
    ArrayDatum kernels;
    if (updateValue<ArrayDatum>(d, nest::kernels, kernels)) {
        for (int i=0; i< kernels.size(); i++) {
            DictionaryDatum kd = getValue< DictionaryDatum >( kernels[i] );
            const std::string kernel_name = getValue<std::string>( kd, nest::name );
            const std::string kernel_params = getValue<std::string>( kd, nest::params );
            addKernel(kernel_name, kernel_params);
        }
    }
}

void H5Neurons::addKernel(const std::string& name, TokenArray params)
{
	if (name == "add") {
		std::vector<float> v(params.size());
		for (int i=0; i<params.size(); i++)
			v[i] = params[i];
		kernel.push_back< kernel_add<float> >(v);
	}
	if (name == "multi") {
		std::vector<float> v(params.size());
		for (int i=0; i<params.size(); i++)
			v[i] = params[i];
		kernel.push_back< kernel_multi<float> >(v);
	}
}

void H5Neurons::import(DictionaryDatum& dout)
{
    int rank = nest::kernel().mpi_manager.get_rank();
    int size = nest::kernel().mpi_manager.get_num_processes();

    H5CellLoader cellLoader(filename);

    const  uint64_t non= cellLoader.getNumberOfCells(neurons_.parameter_names[0]);
    neurons_.resize(non);

    //the id of the fist new created neuron
    nest::index first_neuron = nest::kernel().node_manager.size();

    //loads all parameters for the local neurons based on NEST neuron distribution
    cellLoader.loadLocalParameters(neurons_.parameter_names, non, first_neuron, neurons_);
    GIDCollectionDatum added_neurons = CreateNeurons();
    def< GIDCollectionDatum >( dout, names::added, added_neurons );

    const bool with_subnet = (subnet_name != "");
    if (with_subnet) {
        cellLoader.loadSubnets(non, subnet_name, neurons_);
        def< GIDCollectionDatum >( dout, names::subnet, CreateSubnets(added_neurons) );
    }
}

/*
 * Create subnets for each unique entry inside 'subnet' dataset
 */
GIDCollectionDatum H5Neurons::CreateSubnets(const GIDCollectionDatum& added_neurons)
{
    //find all subnets
    std::vector<int> unique_subnets;
    for (int i=0; i<neurons_.size(); i++) {
        if (!(std::find(unique_subnets.begin(), unique_subnets.end(), neurons_[i].subnet_) != unique_subnets.end())) {
            unique_subnets.push_back(neurons_[i].subnet_);
        }
    }
  
    //del 0 subnet, because 0 subnet means main network
    int n_newSubnets = unique_subnets.size();
    if (std::find(unique_subnets.begin(), unique_subnets.end(), 0)!= unique_subnets.end())
        n_newSubnets--;
    
    
    //for (int i=0; i<n_newSubnets; i++) {
    //return only first gidcollection
    int i=0;
        std::vector<long> gids;
        for (int j=0; j<neurons_.size(); j++)
            if (unique_subnets[i] == neurons_[i].subnet_)
                gids.push_back(added_neurons[j]);
    
    //}
    return GIDCollectionDatum(TokenArray(gids));
}

/*
 * Create Neurons in subnets using loaded parameters
 */
GIDCollectionDatum H5Neurons::CreateNeurons()
{  
    const index non = neurons_.size();
    const index first_neuron_gid = kernel().node_manager.size();
    const index last_neuron_gid = nest::kernel().node_manager.add_node(neurons_.model_id_, non);
    
    GIDCollectionDatum added_neurons(first_neuron_gid, last_neuron_gid);
    
    //set parameters of created neurons
    //ids of neurons are continously even though they might be in different subnets
    for (int i=0;i<non;i++) {
        nest::Node* node = nest::kernel().node_manager.get_node(added_neurons[i]);
        if (nest::kernel().node_manager.is_local_node(node)) {
            DictionaryDatum d( new Dictionary );
            
            std::vector<float> values(neurons_[i].parameter_values_, neurons_[i].parameter_values_+neurons_.parameter_names.size());
            
            //apply kernels
            values = kernel(values);
            
            //copy values into sli data objects
            for (int j=0; j<model_param_names.size(); j++)
                def< double_t >( d, model_param_names[j], values[j] );
            
            //pass sli objects to neuron
            node->set_status(d);
        }
    }
    return added_neurons;
}
