#include "NESTNodeSynapse.h"
#include "nmpi.h"
#include <cstring>

#include "communicator.h"

NESTNodeSynapse::NESTNodeSynapse()
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
  buf[3] = *reinterpret_cast<int*>(&delay);
  buf[4] = *(reinterpret_cast<int*>(&delay)+1);
  buf[5] = *reinterpret_cast<int*>(&weight);
  buf[6] = *(reinterpret_cast<int*>(&weight)+1);
  buf[7] = *reinterpret_cast<int*>(&U0);
  buf[8] = *(reinterpret_cast<int*>(&U0)+1);
  buf[9] = *reinterpret_cast<int*>(&TauRec);
  buf[10] = *(reinterpret_cast<int*>(&TauRec)+1);
  buf[11] = *reinterpret_cast<int*>(&TauFac);
  buf[12] = *(reinterpret_cast<int*>(&TauFac)+1);
}
void NESTNodeSynapse::deserialize(unsigned int* buf)
{
  source_neuron_ = buf[0];
  target_neuron_ = buf[1];
  node_id_ = buf[2];
  
  delay = *reinterpret_cast<double*>(&buf[3]);
  weight = *reinterpret_cast<double*>(&buf[5]);
  U0 = *reinterpret_cast<double*>(&buf[7]);
  TauRec = *reinterpret_cast<double*>(&buf[9]);
  TauFac = *reinterpret_cast<double*>(&buf[11]);
}
bool NESTNodeSynapse::operator<(const NESTNodeSynapse& rhs) const
{
  return node_id_ < rhs.node_id_;
}