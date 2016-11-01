import h5py
import numpy as np

#settings
N= 7500

#create neuron file
fn = h5py.File('neurons.h5')
fn['E_L'] = np.random.uniform(low=-80, high=-60, size=(N,1))
fn['I_e'] = np.random.uniform(low=5.0, high=15.0, size=(N,1))
fn['V_th'] = np.random.uniform(low=-55.0, high=-45.0, size=(N,1))
fn['V_reset'] = np.random.uniform(low=-80.0, high=-60.0, size=(N,1))
fn['C_m'] = np.random.uniform(low=200.0, high=300.0, size=(N,1))
fn['tau_m'] = np.random.uniform(low=9.0, high=11.0, size=(N,1))
fn['tau_syn'] = np.random.uniform(low=1.5, high=2.5, size=(N,1))
fn['t_ref'] = np.random.uniform(low=1.5, high=2.5, size=(N,1))


#create positions
fn['x'] = np.random.uniform(low=0.0, high=1000.0, size=(N,1))
fn['y'] = np.random.uniform(low=0.0, high=1000.0, size=(N,1))
fn['z'] = np.random.uniform(low=0.0, high=1000.0, size=(N,1))

#add tag to a few neurons
fn['tag'] = np.zeros([N,1], dtype=np.int32)
fn['tag'][10:50] = 1

fn.close()
