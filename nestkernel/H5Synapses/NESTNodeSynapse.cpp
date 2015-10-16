#include "NESTNodeSynapse.h"
#include "nmpi.h"
#include <cstring>

#include "communicator.h"

NESTNodeSynapse::NESTNodeSynapse(): num_used_prop_values_(5)
{}
NESTNodeSynapse::NESTNodeSynapse(const unsigned int& source_neuron, const unsigned int& target_neuron)
{
  set(source_neuron, target_neuron);
}
NESTNodeSynapse::~NESTNodeSynapse()
{}
void NESTNodeSynapse::set(const unsigned int& source_neuron, const unsigned int& target_neuron)
{
  source_neuron_ = source_neuron;
  target_neuron_ = target_neuron;
  node_id_ = target_neuron_ % nest::Communicator::get_num_processes();
}
void NESTNodeSynapse::integrateOffset(const int& offset)
{
  source_neuron_ += offset;
  target_neuron_ += offset;
  node_id_ = target_neuron_ % nest::Communicator::get_num_processes();
}
void NESTNodeSynapse::serialize(unsigned int* buf)
{
  buf[0] = source_neuron_;
  buf[1] = target_neuron_;
  buf[2] = node_id_;
  memcpy(&buf[3], &prop_values_[0], num_used_prop_values_*sizeof(double));
}
void NESTNodeSynapse::deserialize(unsigned int* buf)
{
  source_neuron_ = buf[0];
  target_neuron_ = buf[1];
  node_id_ = buf[2];
  
  memcpy(&prop_values_[0], &buf[3], num_used_prop_values_*sizeof(double));
}
bool NESTNodeSynapse::operator<(const NESTNodeSynapse& rhs) const
{
  return node_id_ < rhs.node_id_;
}