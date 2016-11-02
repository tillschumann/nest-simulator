import h5py
import numpy as np

#settings
N=7500
FANOUT=1000

#create file
fs = h5py.File('synapses.h5')

#create neuron dataset
neuron_type = np.dtype([('id', 'u4'),
                      ('syn_n', 'u4'),
                      ('syn_ptr', 'u8')])

buffer=np.empty(N,dtype=neuron_type)
buffer['id'] = np.arange(0,N)	
buffer['syn_n'] = FANOUT * np.ones(N)
buffer['syn_ptr'] = np.cumsum(buffer['syn_n']) - FANOUT
fs['neuron'] = buffer

#create synapse dataset
syn_type = np.dtype([('target', 'u4'),
                      ('delay', 'f4'),
                      ('weight', 'f4'),
		      ('U', 'f4'),
		      ('tau_rec', 'f4'),
                      ('tau_fac', 'f4')])

ds = fs.create_dataset("syn", (N*FANOUT,), dtype=syn_type)


#fill data set iteratively to be not limited by memory

#number of copied values per iteraion
n=10000000

#create dataset
buffer=np.empty(n,dtype=syn_type)
buffer['target'] = np.random.random_integers(0, N-1, n)
buffer['delay'] = np.random.uniform(low=0.5, high=1.0, size=n)
buffer['weight'] = np.random.uniform(low=0.5, high=3.0, size=n)
buffer['U'] = np.random.uniform(low=0.5, high=1.0, size=n)
buffer['tau_rec'] = np.random.uniform(low=700.0, high=900.0, size=n)
buffer['tau_fac'] = np.random.uniform(low=0.0, high=1.0, size=n)

#total number of entries
l = len(ds)

i=0
while i<l:
	#value to be copied
	c = min(i+n,l) - i
	#copy the buffer into the dataset
	ds[i:min(i+n,l)]=buffer[0:c]
	i+=n

fs.close()
