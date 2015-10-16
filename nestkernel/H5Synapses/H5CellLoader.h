#include "hdf5.h"
#include <vector>

class H5CellLoader
{
protected:
  hid_t file_id_, gid_; //hdf5 file pointer
  
  struct H5Dataset
  {
    hid_t dataset_id_;
    
    H5Dataset(const H5CellLoader* loader, const char* datasetname)
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
  
  size_t getNumberOfCells(H5Dataset& cell_dataset)
  {
    hid_t dataspace_id = H5Dget_space(cell_dataset.getId());
    H5Sget_simple_extent_dims(dataspace_id, &count, NULL); // get vector length from x dataset length
    H5Sclose (dataspace_id);
    
    return count;
  }
  
public:
  H5CellLoader(const char* h5file)
  {
    file_id_ = H5Fopen (h5file, H5F_ACC_RDONLY, H5P_DEFAULT);
    gid_ = H5Gopen(file_id_,"/",H5P_DEFAULT);
  }
  ~H5CellLoader()
  {
    H5Gclose(gid_);
    H5Fclose (file_id_);
  }
  
  size_t getNumberOfCells()
  {
    H5Dataset cell_dataset(this,"cells");
    return getNumberOfCells(cell_dataset);
  }
  
  void loadCellProperties(GIDVector<NESTNodeNeuron>& neurons)
  {
    H5Dataset cell_dataset(this,"cells");
    
    
    noc = getNumberOfCells(cell_dataset);
    
    neurons.resize(noc);
    for (int i=0; i<noc;i++)
      neurons.prop_values_.resize(14);
    
    
    std::vector< std::string > prop_names(14);
    
    prop_names[0] = "Delta_T";
    prop_names[1] = "E_L";
    prop_names[2] = "E_ex";
    prop_names[3] = "E_in";
    prop_names[4] = "V_peak";
    prop_names[5] = "V_reset";
    prop_names[6] = "V_th";
    prop_names[7] = "a";
    prop_names[8] = "b";
    prop_names[9] = "x";
    prop_names[10] = "y";
    prop_names[11] = "z";
    
    hid_t memtype = H5Tcreate (H5T_COMPOUND, sizeof (neurons)); // sizeof (neurons) -> make function
    //additionally load following columns
    for (int i=0; i< prop_names.size(); i++) {
      H5Tinsert (memtype, prop_names[i], HOFFSET (NESTNodeNeuron, prop_values_)+i*sizeof(double), H5T_NATIVE_DOUBLE);
    }
    //subnet has to be given too
    H5Tinsert (memtype, "subnet", HOFFSET (NESTNodeNeuron, subnet_), H5T_NATIVE_INT);
    
    //load only neuron parameters which are needed based on NEST internal round robin fashion
    hsize_t dataspace_offset[1] = RANK + neuron_id_offset;
    hsize_t dataspace_stride[1] = NUM_PROCESSES;
    hsize_t count[1] =  noc/NUM_PROCESSES + (noc%NUM_PROCESSES<RANK);
    hsize_t dataspace_block[1] = 1;
    
    dataspace_id = H5Dget_space(cell_dataset.getId);
    H5Sselect_hyperslab (dataspace_id, H5S_SELECT_SET, dataspace_offset, dataspace_stride, count, dataspace_block);
    
    hid_t memspace_id = H5Screate_simple (1, count, NULL);
    
    H5Dread (cell_dataset.getId, memtype, memspace_id, dataspace_id,	H5P_DEFAULT, &neurons[0]);
  
    H5Sclose (memspace_id);
    H5Sclose (dataspace_id);
  } 
};