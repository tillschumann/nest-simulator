#include <algorithm>
#include <numeric>
#include <vector>
#include <string>

#include "h5import/h5interface.h"
#include "h5import/synapse_buffer.h"

#ifndef H5IMPORT_H5READER_CLASS
#define H5IMPORT_H5READER_CLASS

namespace h5import {

class H5SynapseFile : public H5File
{
public:
    /*
     *  store data from neuron link dataset
     */
    struct NeuronLink
    {
      int id;
      int syn_n;
      uint64_t syn_ptr;

      static bool
      	MinSynPtr( const NeuronLink& a, const NeuronLink& b )
      	{
      	return a.syn_ptr < b.syn_ptr;
      	};
    };

protected:

  hid_t memtype_;
  H5Dataset* dataset_ptr_;

  //pointer in syn dataset
  uint64_t global_offset_;
  //considered pointer to last entry
  uint64_t totalsize_;
  //number of data transfered on each read call
  uint64_t transfersize_;
  //number of columns in syn dataset
  uint32_t num_compound_;

  int num_processes_;
  int rank_;

  std::vector< NeuronLink > neuronLinks_;

  /*
   *  return size from dataset
   */
  size_t size( H5Dataset* dataset ) const;

  /*
   *  load neuron links from hdf5 file
   */
  void loadNeuronLinks();

  /*
   *  remove not needed neuron links to reduce memory footprint
   *  function is memory intense
   */
  void removeNotNeededNeuronLinks();

public:
  H5SynapseFile( const std::string& h5file,
              const std::vector< std::string >& datasets,
              const uint64_t& transfersize,
              const uint64_t& limittotalsize = -1 );

    ~H5SynapseFile();

    /*
     * Returns the number of entires in dataset
     */
    inline size_t size()
    {
        return dataset_ptr_->size();
    }

    /*
     * returns of file pointer reached end of file
     */
    inline bool eof() const
    {
        return totalsize_ <= global_offset_;
    }

    /*
     * read block from dataset and move internal pointer forward
     */
    void readblock( SynapseBuffer& synapses, H5View& view );

    /*
    * search source neuron in neuronlinks and integrate them in the synapse list
    */
    void integrateSourceNeurons( SynapseBuffer& synapses, const H5View& view );
};
}; //end of h5import namespace

#endif
