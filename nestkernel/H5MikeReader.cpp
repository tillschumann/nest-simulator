#include <iostream>
#include "H5Synapses/H5Synapses.h"
#include <new>

/*void outOfMem() {  
#pragma omp single
  {
    std::cerr << "Out of memory\t";
    std::cerr << "rank=" << nest::Communicator::get_rank() << "\t";
    std::cerr << H5SynMEMPredictor::instance->toString() << std::endl;
  } 
  throw std::bad_alloc();
   
  //exit(1);
}*/

void H5MikeReader(const std::string& con_dir, const std::string& cell_file, nest::index nest_offset, const Name& synmodel_name, TokenArray synparam_names)
{
  //std::set_new_handler(outOfMem);
  
  //omp_set_dynamic(true);
  
  
  
};