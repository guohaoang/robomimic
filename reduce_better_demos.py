import os
import h5py
import numpy as np
type_to_id = {'worse':[0,0,1], 'okay':[0,1,0], 'better':[1,0,0]}

dirs = ['can',
        'lift',
        'square',
        'transport']
for env in dirs:
    path = "datasets/" + env + "/mh"
    old = path + "/low_dim.hdf5"
    new = path + "/low_dim_fewer_better.hdf5"
    os.system('cp {0} {1}'.format(old, new))

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

def remove_eighty_percent_better(f):
    # Get demos to be deleted
    sorted_better_demos = sorted(f['mask']['better'][:])
    num_remaining_demos = int(len(sorted_better_demos)/5)
    demos_to_be_deleted = sorted_better_demos[num_remaining_demos:]
    print("demos to be deleted", demos_to_be_deleted)
    print("nuber of remaining better demos", num_remaining_demos)

    # Delete demos in masks
    for k in f['mask'].keys():
        original_arr = f['mask'][k][:]
        new_arr = np.array([item for item in original_arr if item not in demos_to_be_deleted])
        del f['mask'][k]
        f['mask'].create_dataset(k, data=new_arr)

    # Delete demos in data
    demos_to_be_deleted_strings = [demo.decode("utf-8") for demo in demos_to_be_deleted]
    for demo in f['data'].keys():
        if demo in demos_to_be_deleted_strings:
            del f['data'][demo]

def remove_demos_without_task_id(f):
    demos_without_task_id = []
    for demo in f['data'].keys():
        if 'task_id' not in f['data'][demo]['obs'].keys():
            demos_without_task_id.append(demo)
    for demo in demos_without_task_id:
        del f['data'][demo]
    for k in f['mask'].keys():
        original_arr = f['mask'][k][:]
        new_arr = np.array([item for item in original_arr if item.decode("utf-8") not in demos_without_task_id])
        del f['mask'][k]
        f['mask'].create_dataset(k, data=new_arr)

for dir in dirs:
    print('modifying ' + dir)
    path = 'datasets/{}/mh/low_dim_fewer_better.hdf5'.format(dir)
    f = h5py.File(path, "r+")
    data=f['data']
    mask=f['mask']

    for k,v in type_to_id.items():
        demos = [demo.decode("utf-8") for demo in mask[k]]
        add_task_ids_for_obs(data, demos, v)
        add_task_ids_for_next_obs(data, demos, v)

    remove_demos_without_task_id(f)
    remove_eighty_percent_better(f)

    f.close()