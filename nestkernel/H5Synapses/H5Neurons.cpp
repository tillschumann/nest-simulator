#include <string>

#include "H5Neurons.h"

#include "network.h"
#include "node.h"
#include "nestmodule.h"
#include "nest_names.h"

#include "H5CellLoader.h"



H5Neurons::H5Neurons(const Name model_name, TokenArray param_names, const Name subnet_name)
{
  for (int i=0; i<param_names.size(); i++) { 
    neurons_.parameter_names.push_back(param_names[i]);
  }
  const Token neuron_model = nest::NestModule::get_network().get_modeldict().lookup(model_name);
  neurons_.model_id_ = static_cast<nest::index>(neuron_model);
  
  neurons_.with_subnet = (subnet_name != "");
  if (neurons_.with_subnet)
    neurons_.subnet_name = subnet_name.toString();
}

void H5Neurons::import(const std::string& filename)
{
  int rank = nest::Communicator::get_rank();
  int size = nest::Communicator::get_num_processes();
  
  H5CellLoader cellLoader(filename);
  
  const  uint64_t numberOfNeurons= cellLoader.getNumberOfCells(neurons_.parameter_names[0]);     

  neurons_.resize(numberOfNeurons);
  
  if (neurons_.with_subnet) {
    cellLoader.loadSubnets(numberOfNeurons, neurons_);
    CreateSubnets();
  }
  
  //the id of the fist new created neuron
  nest::index first_neuron = nest::NestModule::get_network().size();
  
  //loads all parameters for the local neurons based on NEST neuron distribution
  cellLoader.loadLocalParameters(numberOfNeurons, first_neuron, neurons_);
  CreateNeurons();
}

/*
 * Create subnets for each unique entry inside 'subnet' dataset
 */
void H5Neurons::CreateSubnets()
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
  
  
  if (n_newSubnets>0) {
    //create subnets:
    const std::string sub_modname = "subnet"; //
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
  
  //std::cout << "numberOfSubnets=" << n_newSubnets << std::endl;
}

/*void shuffelNeurons()
{
  std::vector< bool > swap_please(neurons_.size());
  for (int i=0; i<neurons_.size(); i++)
    if (swap_please[i])
      neurons_.swap(i,i+1);
}*/

/*
 * Create Neurons in subnets using loaded parameters
 */
void H5Neurons::CreateNeurons()
{  
  const uint32_t non = neurons_.size();
  
  //jump to main network
  nest::index current_subnet=0;
  nest::NestModule::get_network().go_to(current_subnet);

  int last_index=0;
  if (neurons_.with_subnet) {
    for (int i=0;i<non;i++) {
      if (current_subnet!=neurons_[i].subnet_) {
	if (i>last_index) // only the case if first neuron is not 0 subnet
	  nest::NestModule::get_network().add_node(neurons_.model_id_, i-last_index);
	current_subnet=neurons_.getSubnet(i);
	last_index=i;
	
	//jump to subnetwork
	nest::NestModule::get_network().go_to(subnetMap_[neurons_[i].subnet_]);
      }
    }    
  }
  
  //if there are no subnets all neurons are created continously
  //else the command creates the neurons for the last entries with same subnet
  const nest::index last_neuron_id = nest::NestModule::get_network().add_node(neurons_.model_id_, non-last_index);
  const nest::index first_neuron_id = last_neuron_id - non +1;

  //set parameters of created neurons
  //ids of neurons are continously even though they might be in different subnets
  for (int i=0;i<non;i++) {
    nest::Node* node = nest::NestModule::get_network().get_node(first_neuron_id+i);
    if (nest::NestModule::get_network().is_local_node(node))
    {
      DictionaryDatum d( new Dictionary );
      for (int j=0; j<neurons_.parameter_names.size(); j++)
	def< double_t >( d, neurons_.parameter_names[j], neurons_.getParameter(i, j) );
      node->set_status(d);
    }
  }
}