#include "hdf5.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include "mpi.h"

#include "network.h"
#include "nestmodule.h"

#ifndef H5SYNAPSESLOADER_CLASS
#define H5SYNAPSESLOADER_CLASS

struct NeuronLink {
  int id;
  int syn_n;
  uint64_t syn_ptr;
};

struct H5View {
  hsize_t offset[1];
  hsize_t stride[1];
  hsize_t count[1];
  hsize_t block[1];

  H5View(hsize_t icount, hsize_t ioffset=0, hsize_t istride=1, hsize_t iblock=1) {
    offset[0]=ioffset;
    stride[0]=istride;
    count[0]=icount;
    block[0]=iblock;    
  };
  
  inline uint64_t view2dataset(const uint64_t& v_idx) const
  {
    return  offset[0] + (v_idx / block[0]) * (stride[0]-1) + v_idx;
  }
  
  static bool MinSynPtr(const NeuronLink& a,  const NeuronLink& b)
  {
    return a.syn_ptr < b.syn_ptr; 
  };
};

class H5SynapsesLoader
{
protected:
  hid_t file_id_, gid_; //hdf5 file pointer
  
  uint64_t& n_readSynapses;
  uint64_t& n_SynapsesInDatasets;
  
  uint64_t total_num_syns_;
  uint64_t global_offset_;
  
  
  int NUM_PROCESSES;
  int RANK;
 
  std::vector < NeuronLink > neuronLinks;
  
  struct H5Dataset
  {
    hid_t dataset_id_;
    
    H5Dataset(const H5SynapsesLoader* loader, const char* datasetname)
    {
      dataset_id_ = H5Dopen2 (loader->file_id_, datasetname, H5P_DEFAULT);
    }
  
    ~H5Dataset()
    {
      H5Dclose (dataset_id_);
    }
    
    hid_t getId() const
    {
      return dataset_id_;
    }
  };
  
  size_t getNumberOfSynapses(H5Dataset& cell_dataset)
  {
    hsize_t count;
    
    hid_t dataspace_id = H5Dget_space(cell_dataset.getId());
    H5Sget_simple_extent_dims(dataspace_id, &count, NULL); // get vector length from x dataset length
    H5Sclose (dataspace_id);
    
    return count;
  }
  
public:
  H5SynapsesLoader(const std::string h5file, uint64_t& n_readSynapses, uint64_t& n_SynapsesInDatasets) : global_offset_(0), n_readSynapses(n_readSynapses), n_SynapsesInDatasets(n_SynapsesInDatasets)
  {
    MPI_Comm_size(MPI_COMM_WORLD, &NUM_PROCESSES);
    MPI_Comm_rank(MPI_COMM_WORLD, &RANK);
    
    
    hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_id, MPI_COMM_WORLD, MPI_INFO_NULL);
    
    file_id_ = H5Fopen (h5file.c_str(), H5F_ACC_RDONLY, fapl_id);
    
    H5Pclose(fapl_id);
    
    
    gid_ = H5Gopen(file_id_,"/",H5P_DEFAULT);
    
    total_num_syns_ = getNumberOfSynapses();
    
    n_SynapsesInDatasets += total_num_syns_;
    
    loadNeuronLinks();
  }
  ~H5SynapsesLoader()
  {
    H5Gclose(gid_);
    H5Fclose (file_id_);
  }
  
  /*
   * Returns the number of synapses (entries in syn dataset)
   */
  size_t getNumberOfSynapses()
  {
    H5Dataset cell_dataset(this,"syn");
    return getNumberOfSynapses(cell_dataset);
  }
  /*
   * returns of file pointer reached end of file
   */
  bool eof()
  {
    return total_num_syns_ <= global_offset_;
  }
  
  /*
   * Load source neuron ids and store in vector
   */
  
  void loadNeuronLinks()
  {
    H5Dataset neuronLink_dataset(this,"neuron");
    
    hid_t memtype = H5Tcreate (H5T_COMPOUND, sizeof (NeuronLink)); // sizeof (neurons) -> make function
    //additionally load following columns
    
    H5Tinsert (memtype, "id", HOFFSET (NeuronLink, id), H5T_NATIVE_INT);
    H5Tinsert (memtype, "syn_n", HOFFSET (NeuronLink, syn_n), H5T_NATIVE_INT);
    H5Tinsert (memtype, "syn_ptr", HOFFSET (NeuronLink, syn_ptr), H5T_NATIVE_ULLONG);
    
    hid_t dataspace_id = H5Dget_space(neuronLink_dataset.getId());
    
    hsize_t count;
    H5Sget_simple_extent_dims(dataspace_id, &count, NULL);
   
    neuronLinks.resize(count);
    
    hid_t memspace_id = H5Screate_simple (1, &count, NULL);
    
    H5Dread (neuronLink_dataset.getId(), memtype, memspace_id, dataspace_id, H5P_DEFAULT, &neuronLinks[0]);
  
    H5Sclose (memspace_id);
    H5Sclose (dataspace_id);
    
    
    std::stable_sort(neuronLinks.begin(), neuronLinks.end(), H5View::MinSynPtr);
  }
 
  /**
   * search source neuron
   * 
   * could be optimized using a binary search alg
   */
  void integrateSourceNeurons(NESTSynapseList& synapses, const H5View& view)
  {
    uint64_t index = view.view2dataset(0);
    
    std::vector < NeuronLink >::const_iterator it_neuronLinks=neuronLinks.begin();
      
    int source_neuron=it_neuronLinks->id;  
    int next_ptr=it_neuronLinks->syn_ptr + it_neuronLinks->syn_n;
    
    for (int i=0; i<synapses.size(); i++)
    {      
      index = view.view2dataset(i);
      while (index>=next_ptr && it_neuronLinks<neuronLinks.end())
      {
	it_neuronLinks++;
	if (it_neuronLinks==neuronLinks.end()) {
	  nest::NestModule::get_network().message( SLIInterpreter::M_ERROR, "H5SynapsesLoader::integrateSourceNeurons()", "HDF5 neuron and synapse dataset dont match");
	  break;
	}
	else {
	  source_neuron=it_neuronLinks->id;
	  next_ptr=it_neuronLinks->syn_ptr+it_neuronLinks->syn_n;
	}
      }
      synapses[i].source_neuron_ = source_neuron;
    }
  }
  
  /*
   * Get num_syns synapses from datasets collectivly
   * Move file pointer to for next function call
   * 
   */
  void iterateOverSynapsesFromFiles(NESTSynapseList& synapses, const uint64_t& num_syns)
  {    
    hid_t memtype = H5Tcreate (H5T_COMPOUND, sizeof (NESTNodeSynapse)); // sizeof (synapses) -> make function
    //target is always given
    H5Tinsert (memtype, "target", HOFFSET (NESTNodeSynapse, target_neuron_), H5T_NATIVE_INT);
    //additionally load following columns
    for (int i=0; i< synapses.prop_names.size(); i++) {
      H5Tinsert (memtype, synapses.prop_names[i].c_str(), HOFFSET (NESTNodeSynapse, prop_values_)+i*sizeof(double), H5T_NATIVE_DOUBLE);
    }
    
    std::vector<uint64_t> global_num_syns(NUM_PROCESSES);
    uint64_t private_num_syns = num_syns;
    
    MPI_Allgather(&private_num_syns, 1, MPI_UNSIGNED_LONG_LONG, &global_num_syns[0], 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
    uint64_t private_offset = std::accumulate(global_num_syns.begin(), global_num_syns.begin()+RANK, global_offset_);
    global_offset_ = std::accumulate(global_num_syns.begin()+RANK, global_num_syns.end(), private_offset); // for next iteration
    
    //load only neuron parameters which are needed based on NEST internal round robin fashion
  
    int64_t count = std::min((int64_t)num_syns, ((int64_t)total_num_syns_-(int64_t)private_offset));
    if (count<0)
      count=0;
    H5View dataspace_view(count, private_offset); 
    
    H5Dataset syn_dataset(this, "syn");
    
    hid_t dataspace_id = H5Dget_space(syn_dataset.getId());
    hid_t memspace_id;
    
    //be careful if there are no synapses to load
    if (dataspace_view.count[0]>0) {
      H5Sselect_hyperslab (dataspace_id, H5S_SELECT_SET, dataspace_view.offset, dataspace_view.stride, dataspace_view.count, dataspace_view.block);
      memspace_id = H5Screate_simple (1, dataspace_view.count, NULL);
    }
    else {
      H5Sselect_none(dataspace_id);
      memspace_id=H5Scopy(dataspace_id);
      H5Sselect_none(memspace_id);
    } 
    
    synapses.resize(dataspace_view.count[0]);
    
    
    // setup collective read operation
    hid_t dxpl_id_ = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(dxpl_id_, H5FD_MPIO_COLLECTIVE);
    
    H5Dread (syn_dataset.getId(), memtype, memspace_id, dataspace_id, dxpl_id_, &synapses[0]);
    
    H5Pclose(dxpl_id_);
  
    H5Sclose (memspace_id);
    H5Sclose (dataspace_id);
    
    // integrate NEST neuron id offset to synapses
    integrateSourceNeurons(synapses, dataspace_view);
    
    // observer variable
    n_readSynapses += dataspace_view.count[0];
  }
};

#endif