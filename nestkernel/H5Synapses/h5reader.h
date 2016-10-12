#include <algorithm>
#include <numeric>
#include <vector>
#include <string>

#include "SynapseList.h"
#include "h5file.h"

#ifndef H5IMPORT_H5READER_CLASS
#define H5IMPORT_H5READER_CLASS

namespace h5import {

class h5reader : public h5file
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
  h5dataset* dataset_ptr_;

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
  size_t size( h5dataset* dataset ) const;

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
    h5reader( const std::string& h5file,
              const std::vector< std::string >& datasets,
              const uint64_t& transfersize,
              const uint64_t& limittotalsize = -1 );

    ~h5reader();

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
    void readblock( SynapseList& synapses, h5view& dataspace_view );

    /*
    * search source neuron in neuronlinks and integrate them in the synapse list
    */
    void integrateSourceNeurons( SynapseList& synapses, const h5view& view );
};
}; //end of h5import namespace

#endif
