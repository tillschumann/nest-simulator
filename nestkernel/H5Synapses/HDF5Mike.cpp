#include "HDF5Mike.h"
#include "NESTNodeSynapse.h"
#include "nmpi.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include <unistd.h>

#include <cstdlib>


/*function... might want it in some class?*/
int HDF5Mike::getdir (const std::string dir, std::vector<SFile_>& files)
{
    DIR *dp;
    struct stat file_stats;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        std::cerr << "Error(" << errno << ") opening " << dir << std::endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
	std::string tmp(dirp->d_name);
	if (tmp.find(".h5")!=std::string::npos) {
	  SFile_ tmp_file;
	  tmp_file.name = dir+"/"+tmp;
	  if (stat(tmp_file.name.c_str(), &file_stats)!=-1)
	    tmp_file.size = (uint64_t)file_stats.st_size;
	  else
	    tmp_file.size = 999;
	  
	  files.push_back(tmp_file);
	}
    }
    closedir(dp);
    return 0;
}

int HDF5Mike::getNumberOfNeurons(const char* coord_file_name)
{  
  herr_t      coord_status_;
  
  hid_t coord_file_id_ = H5Fopen (coord_file_name, H5F_ACC_RDONLY, H5P_DEFAULT);
  hid_t dset = H5Dopen2 (coord_file_id_, "x", H5P_DEFAULT);
  
  hsize_t dims[2];
  hid_t dspace = H5Dget_space(dset);
  H5Sget_simple_extent_dims(dspace, dims, NULL);
  
  H5Sclose(dspace);

  
  coord_status_ = H5Dclose (dset);
  coord_status_ = H5Fclose (coord_file_id_);
  
  return dims[0];
}


void HDF5Mike::loadAllNeuronCoords(const char* coord_file_name, const uint32_t& numberOfNeurons, std::vector<Coords>& neurons_pos)
{
  
  const uint32_t n = numberOfNeurons;
  neurons_pos.resize(n);

  herr_t      coord_status_;
  
  // target neuron ids are based on neuron_id MOD numberOfNodes:
  // So the target neuron ids are: Node_id, Node_id+numberOfNodes, Node_id+2*numberOfNodes, ..
  
  
  hsize_t count[2]  = {n*3,1};
  hsize_t block[2] = {1,1};
    
  hid_t coord_file_id_ = H5Fopen (coord_file_name, H5F_ACC_RDONLY, H5P_DEFAULT);
  hid_t memspace_id = H5Screate_simple (2, count, NULL); 
  
  
  
  
  // load x values
  hid_t dset = H5Dopen2 (coord_file_id_, "x", H5P_DEFAULT);
  //hid_t dataspace_id = H5Dget_space (dset);
  
  //adapt values to x value position vector
   hsize_t offset[2], stride[2];

  offset[0] = 0;//offsetof(struct Coords, x_);
  offset[1] = 0;
  stride[0] = 3;
  stride[1] = 1;
  
  count[0] = n;
  
  //hid_t VCoordsDataType = H5Tcopy(H5T_NATIVE_DOUBLE);
  //H5Tset_size(VCoordsDataType, sizeof(Coords));
  //H5Tset_offset(VCoordsDataType, offsetof(struct Coords, x_));
  
  coord_status_ = H5Sselect_hyperslab (memspace_id, H5S_SELECT_SET, offset,
				stride, count, block);
  coord_status_ = H5Dread (dset, H5T_NATIVE_DOUBLE, memspace_id, H5S_ALL,
		      H5P_DEFAULT, &neurons_pos[0]);
  coord_status_ = H5Dclose (dset);
  
  // load y values
  dset = H5Dopen2 (coord_file_id_, "y", H5P_DEFAULT);
  //dataspace_id = H5Dget_space (dset);  
  //adapt values to y value position vector
  offset[0] = 1;
  //H5Tset_offset(VCoordsDataType, offsetof(struct Coords, y_));
  
  
  coord_status_ = H5Sselect_hyperslab (memspace_id, H5S_SELECT_SET, offset,
				stride, count, block);
  coord_status_ = H5Dread (dset, H5T_NATIVE_DOUBLE, memspace_id, H5S_ALL,
		      H5P_DEFAULT, &(neurons_pos[0]));
  coord_status_ = H5Dclose (dset);
  
  // load z values
  dset = H5Dopen2 (coord_file_id_, "z", H5P_DEFAULT);
  //dataspace_id = H5Dget_space (dset);
  
  //adapt values to z value position vector
  offset[0] = 2;
  
  //H5Tset_offset(VCoordsDataType, offsetof(struct Coords, z_));
  coord_status_ = H5Sselect_hyperslab (memspace_id, H5S_SELECT_SET, offset,
				stride, count, block);
  coord_status_ = H5Dread (dset, H5T_NATIVE_DOUBLE, memspace_id, H5S_ALL,
		      H5P_DEFAULT, &neurons_pos[0]);
  coord_status_ = H5Dclose (dset);
  
  //H5Tclose(VCoordsDataType);
  coord_status_ = H5Sclose (memspace_id);
  coord_status_ = H5Fclose (coord_file_id_);
}

void HDF5Mike::loadSourceNeuonIds(const hid_t& dataset_id, const uint32_t& n)
{
  hsize_t offset[2] = {0,0};
  hsize_t count[2]  = {n,1};
  hsize_t stride[2] = {1,1};
  hsize_t block[2] = {1,1}; 
  
  hid_t memspace_id = H5Screate_simple (2, count, NULL); 
  
  hid_t dataspace_id = H5Dget_space (dataset_id);
  status_ = H5Sselect_hyperslab (dataspace_id, H5S_SELECT_SET, offset,
				stride, count, block);

  
  status_ = H5Dread (dataset_id, H5T_NATIVE_UINT, memspace_id, dataspace_id,
		      H5P_DEFAULT, &buffer_source_neurons[0]);
  
  status_ = H5Sclose (memspace_id);
  status_ = H5Sclose (dataspace_id);
}

void HDF5Mike::loadDataset2Buffers(const uint32_t& i_datasets)
{  
  
  uint32_t& ntarget= number_target_neurons;
  uint32_t& nsource= number_source_neurons;
  uint32_t& nparams = number_params;
  
  char datasetname[256];
  H5Gget_objname_by_idx(gid_, (hsize_t)(i_datasets*2), datasetname, 255 ); //critical selection of dataset
  
  
  hid_t dset = openDataset(datasetname);
  hid_t dataspace_id = H5Dget_space(dset);
  hsize_t dims[3];
  H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
  
  nsource = dims[0];
  ntarget = dims[1];
  nparams = dims[2];
  
  if (buffer_source_neurons.size()<nsource)
    buffer_source_neurons.resize(nsource);
  if (buffer_target_neurons.size()<nsource*ntarget*nparams)
    buffer_target_neurons.resize(nsource*ntarget*nparams);
  
  
  n_SynapsesInDatasets += nsource*ntarget;
  
  H5Gget_objname_by_idx(gid_, (hsize_t)(i_datasets*2+1), datasetname, 255 ); //critical selection of dataset
  hid_t dset_source = openDataset(datasetname);
  loadSourceNeuonIds(dset_source, nsource);
  closeDataSet(dset_source);
  
  hsize_t offset[3] = {0,0,0};
  hsize_t count[3]  = {nsource,ntarget,nparams};
  hsize_t stride[3] = {1,1,1};
  hsize_t block[3] = {1,1,1}; 
  
  hid_t memspace_id = H5Screate_simple (3, count, NULL); 
  
  status_ = H5Sselect_hyperslab (dataspace_id, H5S_SELECT_SET, offset,
				stride, count, block);

  
  status_ = H5Dread (dset, H5T_NATIVE_FLOAT, memspace_id, dataspace_id,
		      H5P_DEFAULT, &buffer_target_neurons[0]);
  
  status_ = H5Sclose (memspace_id);
  status_ = H5Sclose (dataspace_id);
      
  closeDataSet(dset);
}

void HDF5Mike::openFile(const std::string& filename)
{
  file_id_ = H5Fopen (filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  
  gid_ = H5Gopen(file_id_,"/",H5P_DEFAULT);
  
  hsize_t num_obj;
  H5Gget_num_objs(gid_, &num_obj);
  number_datasets = num_obj/2;
}

void HDF5Mike::closeFile()
{
  status_ = H5Gclose(gid_);
  status_ = H5Fclose (file_id_);
}


bool HDF5Mike::endOfMikeFiles()
{
  return (i_hdf5files>=(int)hdf5files.size());
}

/************************************************************************************************************/
/************************************************************************************************************/

HDF5Mike::HDF5Mike(const std::string& con_dir, uint64_t& n_readSynapses, uint64_t& n_SynapsesInDatasets): open_dataset(false), n_readSynapses(n_readSynapses), n_SynapsesInDatasets(n_SynapsesInDatasets)
{
  rank = nest::Communicator::get_rank();
  size = nest::Communicator::get_num_processes();
  
  getdir(con_dir, hdf5files);
  
  preLoadBalancing();
  
  /*if (rank==0) {
    for (uint32_t i=0;i<hdf5files.size(); i++) {
      std::cout << hdf5files[i].name << "\t size=" << hdf5files[i].size << std::endl;
    }
  }*/
  
  i_hdf5files=0; // distribute files between nodes based on mod function 
  i_datasets=0;
  i_target=0;
  i_source=0;
  
  
  number_target_neurons=0;
  number_source_neurons=0;
  number_params=0;

  // init can be done here
  
}

/************************************************************************************************************/

HDF5Mike::~HDF5Mike()
{
}

/************************************************************************************************************/

void HDF5Mike::iterateOverAllSynapsesFromCurrentFile(std::deque<NESTNodeSynapse>& synapses)
{
  openFile(hdf5files[i_hdf5files].name); // fills number_datasets and opens file
  
  for (uint32_t i_datasets = 0; i_datasets < number_datasets; i_datasets++) // iteration over all datasets in file
  {
    
    loadDataset2Buffers(i_datasets); // write buffer_source_neurons_, buffer_target_neurons_, number_target_neurons, number_source_neurons
   
    
    for (uint32_t i_target=0; i_target<number_target_neurons; i_target++) {   // iteration over all synapses
      for (uint32_t i_source=0; i_source<number_source_neurons; i_source++) {
	synapses.push_back(NESTNodeSynapse(buffer_source_neurons[i_source]+1, buffer_target_neurons[i_target*number_source_neurons+i_source]+1)); // +1 because of NEST id offset
	n_readSynapses++;
      }
    }
  }  
  closeFile();
  
  // distribute files between nodes based on mod function 
  
  i_hdf5files++; // size contains nest::Communicator::get_num_processes()
}

/************************************************************************************************************/

void HDF5Mike::iterateOverSynapsesFromFiles(std::deque<NESTNodeSynapse>& synapses, const uint64_t& number_of_synapses)
{
  uint64_t new_synapses=0;
  
  for (;i_hdf5files<hdf5files.size(); i_hdf5files++)
  {
    if (i_datasets==0) {
      openFile(hdf5files[i_hdf5files].name); // fills number_datasets and opens file
    }
    for (;i_datasets < number_datasets; i_datasets++) // iteration over all datasets in file
    {
      if (i_source==0 && i_target==0) {
	loadDataset2Buffers(i_datasets); // write buffer_source_neurons_, buffer_target_neurons_, number_target_neurons, number_source_neurons
      }
      for (;i_source<number_source_neurons; i_source++) {   // iteration over all synapses
	for (;i_target<number_target_neurons; i_target++) {
	  NESTNodeSynapse syn(buffer_source_neurons[i_source], *reinterpret_cast<unsigned int*>(&buffer_target_neurons[(i_source*number_target_neurons+i_target)*number_params]));
	  syn.delay=	buffer_target_neurons[(i_source*number_target_neurons+i_target)*number_params+1];
	  syn.weight=	buffer_target_neurons[(i_source*number_target_neurons+i_target)*number_params+2];
	  syn.U0=	buffer_target_neurons[(i_source*number_target_neurons+i_target)*number_params+3];
	  syn.TauRec=	buffer_target_neurons[(i_source*number_target_neurons+i_target)*number_params+4];
	  syn.TauFac=	buffer_target_neurons[(i_source*number_target_neurons+i_target)*number_params+5];
	  synapses.push_back(syn); // +1 because of NEST id offset
	  new_synapses++;
	  if (new_synapses>=number_of_synapses)
	  {
	    n_readSynapses+=new_synapses;
	    i_target++; // iteration completed, but return will jump over last for loop var increment
	    return; // limit reached
	  }
	}
	i_target=0;
      }
      i_source=0;
    }
    closeFile();
    i_datasets=0;
  }
  // if all files are processed 
  n_readSynapses+=new_synapses;
}


/************************************************************************************************************/

void HDF5Mike::preLoadBalancing()
{
  
  rank = nest::Communicator::get_rank();
  size = nest::Communicator::get_num_processes();
  
  
  //optimize file distribution
  std::sort(hdf5files.begin(), hdf5files.end());
  std::reverse(hdf5files.begin(), hdf5files.end());
  
  std::vector<SFile_> own_files;
  
  //distribute files in modulos fashion
  for (int i=0;i<hdf5files.size();i++)
  {
    if (i%size == rank)
      own_files.push_back(hdf5files[i]);
  }
  
  
  std::vector<uint64_t> load_per_node(size, 0);
 
  uint64_t load_total = 0;
  //load imbalance
  for (uint32_t i=0;i<hdf5files.size();i++)
  {
    load_total += hdf5files[i].size;
    load_per_node[i%size] += hdf5files[i].size;
  }
  
  const uint64_t load_avg = load_total/size;
  
  uint64_t inbalance_total = 0;
  uint64_t inbalance_max = 0;
  uint64_t inbalance_min = load_total;
  for (uint32_t i=0;i<size;i++)
  {
    inbalance_total += abs(load_avg-load_per_node[i]);
    inbalance_min = std::min(inbalance_min, load_per_node[i]);
    inbalance_max = std::max(inbalance_max, load_per_node[i]);
  }
  
  std::cout << "with hdf5 load balancing\trank=" << rank << "\tinbalance_total=" << inbalance_total << "\tinbalance_delta=" << inbalance_max-inbalance_min << "\n";
  
  
  hdf5files.swap(own_files);
  
  
  std::cout << "preLoadBalancing\trank= " << rank << "\tnum_files=" << hdf5files.size() << "\n";
  
  // experimental optimization
  //std::reverse(hdf5files.begin()+1024, hdf5files.end());
}


void HDF5Mike::getValueFromDataset(hid_t& cell_file_id,hid_t& memspace_id,const int& n, const int& offset_i, const char* name, const hid_t& mem_type_id, NESTNodeNeuron* ptr)
{
  hsize_t count[2]  = {n,1};
  hsize_t offset[2], stride[2];
  hsize_t block[2] = {1,1};

  offset[0] = offset_i;//offsetof(struct Coords, x_);
  offset[1] = 0;
  stride[0] = 14;
  stride[1] = 1;
  
  hid_t dset = H5Dopen2 (cell_file_id, name, H5P_DEFAULT);
  
  H5Sselect_hyperslab (memspace_id, H5S_SELECT_SET, offset,
				stride, count, block);
  H5Dread (dset, mem_type_id, memspace_id, H5S_ALL,
		      H5P_DEFAULT, ptr);
  H5Dclose (dset);
}

void HDF5Mike::loadAllNeurons(const char* cell_file_name, const uint32_t& numberOfNeurons, GIDVector<NESTNodeNeuron>& neurons)
{
  const uint32_t n = numberOfNeurons;
  neurons.resize(n);

  herr_t      coord_status_;
  
  // target neuron ids are based on neuron_id MOD numberOfNodes:
  // So the target neuron ids are: Node_id, Node_id+numberOfNodes, Node_id+2*numberOfNodes, ..
  
  hsize_t count[2]  = {n*14,1}; 
  
    
  hid_t cell_file_id = H5Fopen (cell_file_name, H5F_ACC_RDONLY, H5P_DEFAULT);
  hid_t memspace_id = H5Screate_simple (2, count, NULL); 
  
  getValueFromDataset(cell_file_id,memspace_id,n,0,"C_m",H5T_NATIVE_FLOAT, &(neurons[0]));
  getValueFromDataset(cell_file_id,memspace_id,n,1,"Delta_T",H5T_NATIVE_FLOAT, &(neurons[0]));
  getValueFromDataset(cell_file_id,memspace_id,n,2,"E_L",H5T_NATIVE_FLOAT, &(neurons[0]));
  getValueFromDataset(cell_file_id,memspace_id,n,3,"E_ex",H5T_NATIVE_FLOAT, &(neurons[0]));
  getValueFromDataset(cell_file_id,memspace_id,n,4,"E_in",H5T_NATIVE_FLOAT, &(neurons[0]));
  getValueFromDataset(cell_file_id,memspace_id,n,5,"V_peak",H5T_NATIVE_FLOAT, &(neurons[0]));
  getValueFromDataset(cell_file_id,memspace_id,n,6,"V_reset",H5T_NATIVE_FLOAT, &(neurons[0]));
  getValueFromDataset(cell_file_id,memspace_id,n,7,"V_th",H5T_NATIVE_FLOAT, &(neurons[0]));
  getValueFromDataset(cell_file_id,memspace_id,n,8,"a",H5T_NATIVE_FLOAT, &(neurons[0]));
  getValueFromDataset(cell_file_id,memspace_id,n,9,"b",H5T_NATIVE_FLOAT, &(neurons[0]));
  getValueFromDataset(cell_file_id,memspace_id,n,10,"x",H5T_NATIVE_FLOAT, &(neurons[0]));
  getValueFromDataset(cell_file_id,memspace_id,n,11,"y",H5T_NATIVE_FLOAT, &(neurons[0]));
  getValueFromDataset(cell_file_id,memspace_id,n,12,"z",H5T_NATIVE_FLOAT, &(neurons[0]));
  getValueFromDataset(cell_file_id,memspace_id,n,13,"subnet",H5T_NATIVE_INT, &(neurons[0]));

  
  //H5Tclose(VCoordsDataType);
  coord_status_ = H5Sclose (memspace_id);
  coord_status_ = H5Fclose (cell_file_id);
}
