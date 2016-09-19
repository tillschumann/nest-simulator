//#include "nmpi.h"
#include <map>
#include <omp.h>

#include "nest_types.h"
#include "nest_datums.h"
#include "name.h"
#include "tokenarray.h"
#include "kernels.h"

#include "NeuronList.h"

using namespace nest;

class H5Neurons
{
private:
	omp_lock_t tokenLock_;

	NeuronList neurons_;
	kernel_combi<float> kernel_;

	std::string filename_;
	std::vector< std::string > model_param_names_;

	GIDCollectionDatum CreateSubnets(const GIDCollectionDatum& added_neurons);
	GIDCollectionDatum CreateNeurons();
  
public:
	H5Neurons(const DictionaryDatum& din);
	void import(DictionaryDatum& dout);
	void addKernel(const std::string& name, TokenArray params);
};
