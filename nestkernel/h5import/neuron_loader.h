//#include "nmpi.h"
#include <map>
#include <omp.h>

#include "nest_types.h"
#include "nest_datums.h"
#include "name.h"
#include "tokenarray.h"
#include "kernels.h"

#include "h5import/neuron_buffer.h"

using namespace nest;
using namespace h5import;

class NeuronLoader
{
private:
	omp_lock_t tokenLock_;

	NeuronBuffer neurons_;
	kernel_combi<float> kernel_;

	std::string filename_;
	std::vector< std::string > model_param_names_;

	int subnet_key_;

	GIDCollectionDatum CreateSubnets(const GIDCollectionDatum& added_neurons);
	GIDCollectionDatum CreateNeurons();
  
public:
	NeuronLoader(const DictionaryDatum& din);
	void execute(DictionaryDatum& dout);
	void addKernel(const std::string& name, TokenArray params);
};
