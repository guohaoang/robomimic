import h5py
import numpy as np
type_to_id = {'worse':[0,0,1], 'okay':[0,1,0], 'better':[1,0,0]}

path = 'datasets/can/mh/low_dim_task_id_cp.hdf5'
f = h5py.File(path, "r+")
data=f['data']
mask=f['mask']

def add_task_ids_for_obs(data, keys, task_id):
    for k in keys:
        demo = data[k]
        obs = demo['obs']
        t = obs['object'].shape[0]
        task_indices_arr = np.tile(np.array(task_id), (t, 1))
        obs.create_dataset("task_id", shape=(t, len(task_id)), dtype='f8', data=task_indices_arr)
def add_task_ids_for_next_obs(data, keys, task_id):
    for k in keys:
        demo = data[k]
        obs = demo['next_obs']
        t = obs['object'].shape[0]
        task_indices_arr = np.tile(np.array(task_id), (t, 1))
        obs.create_dataset("task_id", shape=(t, len(task_id)), dtype='f8', data=task_indices_arr)

for k,v in type_to_id.items():
    demos = [demo.decode("utf-8") for demo in mask[k]]
    add_task_ids_for_obs(data, demos, v)
    add_task_ids_for_next_obs(data, demos, v)

f.close()