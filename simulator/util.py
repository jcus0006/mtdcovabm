import datetime
import numpy as np
import scipy.stats as stats
import powerlaw
import matplotlib.pyplot as plt
import random
import bisect
import multiprocessing.shared_memory as shm
from cells_mp import CellType

def day_of_year_to_day_of_week(day_of_year, year):
    date = datetime.datetime(year, 1, 1) + datetime.timedelta(day_of_year - 1)
    weekdaystr = date.strftime("%A")
    weekday = -1
    match weekdaystr:
        case "Monday":
            weekday = 1
        case "Tuesday":
            weekday = 2
        case "Wednesday":
            weekday = 3
        case "Thursday":
            weekday = 4
        case "Friday":
            weekday = 5
        case "Saturday":
            weekday = 6
        case "Sunday":
            weekday = 7

    return weekday, weekdaystr

def sample_gamma(gamma_shape, min, max, k = 1, returnInt = False):
    if min == max:
        return min
    
    gamma_scale = (max - min) / (gamma_shape * k)

    sample = np.random.gamma(gamma_shape, gamma_scale)

    if returnInt:
        return round(sample)
    
    return sample

def sample_gamma_reject_out_of_range(gamma_shape, min, max, k = 1, returnInt = False, useNp = False):
    if useNp:
        sample = min - 1

        while sample < min or sample > max:
            sample = sample_gamma(gamma_shape, min, max, k, returnInt)
    else:
        scale = (max - min) / (gamma_shape * k)

        trunc_gamma = stats.truncnorm((min - scale) / np.sqrt(gamma_shape),
                              (max - scale) / np.sqrt(gamma_shape),
                              loc=scale, scale=np.sqrt(gamma_shape))
        
        sample = trunc_gamma.rvs()

    return sample

# Generate a variable number of group_sizes in the range min_size to max_size.
# Sample from a gamma distribution. Default gamma params favour smaller values (use gamma_shape > 1 to favour larger values)
def random_group_partition(group_size, min_size, max_size, gamma_shape = 0.5, k=1):
    group_sizes = []
    
    while sum(group_sizes) < group_size:
        sampled_group_size = sample_gamma_reject_out_of_range(gamma_shape, min_size, max_size, k, True, True)

        if sum(group_sizes) + sampled_group_size > group_size:
            sampled_group_size = group_size - sum(group_sizes)

        group_sizes.append(sampled_group_size)
    
    return group_sizes

def sample_log_normal(mean, std, size, isInt=False):
    dist_mean  = np.log(mean**2 / np.sqrt(std**2 + mean**2)) # Computes the mean of the underlying normal distribution
    dist_sigma = np.sqrt(np.log(std**2/mean**2 + 1)) # Computes sigma for the underlying normal distribution
    samples = np.random.lognormal(mean=dist_mean, sigma=dist_sigma, size=size)

    if isInt:
        int_samples = []
        for i in range(len(samples)):
            sample = samples[i]
            int_samples.append(round(sample))

        if size == 1:
            return int_samples[0]
        else:
            return int_samples

    if size == 1:
        return samples[0]
    
    return samples

def set_age_brackets(agent, agent_uid, age_brackets, age_brackets_workingages, agents_ids_by_agebrackets, set_working_age_bracket=True):
    age = agent["age"]
    # agents_ids_by_ages[agent_uid] = agent["age"]

    agent["age_bracket_index"] = -1
    for i, ab in enumerate(age_brackets):
        if age >= ab[0] and age <= ab[1]:
            agent["age_bracket_index"] = i
            
            agents_ids_by_agebrackets[i].append(agent_uid)

            break
    
    if set_working_age_bracket:
        agent["working_age_bracket_index"] = -1
        for i, ab in enumerate(age_brackets_workingages):
            if age >= ab[0] and age <= ab[1]:
                agent["working_age_bracket_index"] = i
                break

    return agent, age, agents_ids_by_agebrackets

def generate_sociability_rate_powerlaw_dist(temp_agents, agents_ids_by_agebrackets, powerlaw_distribution_parameters, params, sociability_rate_min, sociability_rate_max, figure_count):
    for agebracket_index, agents_ids_in_bracket in agents_ids_by_agebrackets.items():
        powerlaw_dist_params = powerlaw_distribution_parameters[agebracket_index]

        exponent, xmin = powerlaw_dist_params[2], powerlaw_dist_params[3]
        # exponent, xmin = 2.64, 4.0
        dist = powerlaw.Power_Law(xmin=xmin, parameters=[exponent])

        if len(agents_ids_in_bracket) > 0:
            if len(agents_ids_in_bracket) > 1:
                agents_contact_propensity = dist.generate_random(len(agents_ids_in_bracket))

                min_arr = np.min(agents_contact_propensity)
                max_arr = np.max(agents_contact_propensity)

                normalized_arr = (agents_contact_propensity - min_arr) / (max_arr - min_arr) * (sociability_rate_max - sociability_rate_min) + sociability_rate_min

                if params["visualise"]:
                    figure_count += 1
                    plt.figure(figure_count)
                    bins = np.logspace(np.log10(min(agents_contact_propensity)), np.log10(max(agents_contact_propensity)), 50)
                    plt.hist(agents_contact_propensity, bins=bins, density=True)
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.xlabel('Value')
                    plt.ylabel('Frequency')
                    plt.show(block=False)

                for index, agent_id in enumerate(agents_ids_in_bracket):
                    agent = temp_agents[agent_id]

                    agent["soc_rate"] = normalized_arr[index]
            else: # this will never be hit, but in a single case, would favour the lower end of the range with a gamma dist
                single_agent_id = agents_ids_in_bracket[0]

                agent = temp_agents[single_agent_id]             

                agent["soc_rate"] = sample_gamma_reject_out_of_range(0.5, sociability_rate_min, sociability_rate_max, useNp=True)

    return temp_agents

def set_public_transport_regular(agent, usage_probability):
    public_transport_rand = random.random()

    if public_transport_rand < usage_probability:
        agent["pub_transp_reg"] = True
    else:
        agent["pub_transp_reg"] = False

    return agent

def convert_celltype_to_simcelltype(cellid, cells=None, celltype=None):
    if cells is not None and celltype is None:
        cell = cells[cellid]
        celltype = cell["type"]

    match celltype:
        case "household":
            return "residence"
        case "workplace":
            return "workplace"
        case "accom":
            return "residence"
        case "hospital":
            return "community"
        case "entertainment":
            return "community"
        case "school":
            return "school"
        case "institution":
            return "residence"
        case "transport":
            return "community"
        case "religion":
            return "community"
        case "airport":
            return "community"
        case _:
            return "community"
        
def convert_celltype_str_to_enum(celltype):
    match celltype:
        case "household":
            return CellType.Household
        case "workplace":
            return CellType.Workplace
        case "accom":
            return CellType.Accommodation
        case "hospital":
            return CellType.Hospital
        case "entertainment":
            return CellType.Entertainment
        case "school":
            return CellType.School
        case "institution":
            return CellType.Institution
        case "transport":
            return CellType.Transport
        case "religion":
            return CellType.Religion
        case "airport":
            return CellType.Airport
        case _:
            return CellType.Undefined
        
# takes the contact structure in agents_contacts
# can be used for both potential contacts and direct contacts (as they bear the same structure)
# if agents_degree is passed (not none), it will only return the contact id if the relevant degree is greater than 0
# for direct cotacts, agents_degrees may be left out
def get_all_contacts_ids_by_id(id, agents_contacts_keys, agents_ids = None, agents_degrees = None, shuffle=False):
    contacts_ids = []
    for pairid in agents_contacts_keys:
        if id in pairid:
            contact_id = pairid[1] if pairid[0] == id else pairid[0]

            if agents_ids is not None and agents_degrees is not None:
                contact_index = agents_ids.index(contact_id)
                contact_degree = agents_degrees[contact_index]

                if contact_degree > 0:
                    contacts_ids.append(contact_id)
            else:
                contacts_ids.append(contact_id)
            
    if len(contacts_ids) == 0:
        return None
    
    contacts_ids = np.array(contacts_ids)

    if shuffle:
        np.random.shuffle(contacts_ids)

    return contacts_ids

def get_all_contacts_ids_by_id_and_timesteprange(id, agents_contacts_keys, traced_ts, min_ts, max_ts, agents_ids = None, agents_degrees = None, shuffle=False):
    contacts_ids = []

    if max_ts is None:
        max_ts = traced_ts # when contact tracing is done, start from that timestep until 0
    
    if min_ts is None:
        min_ts = traced_ts # when contact tracing is traced for e.g. 24 hours, trace until the same timestep on X prev days

    for pairid, (st_ts, _) in agents_contacts_keys:
        if (id in pairid and 
            (st_ts >= min_ts) and 
            (st_ts <= max_ts)):
            contact_id = pairid[1] if pairid[0] == id else pairid[0]

            if agents_ids is not None and agents_degrees is not None:
                contact_index = agents_ids.index(contact_id)
                contact_degree = agents_degrees[contact_index]

                if contact_degree > 0:
                    contacts_ids.append(contact_id)
            else:
                contacts_ids.append(contact_id)
            
    if len(contacts_ids) == 0:
        return None
    
    contacts_ids = np.array(contacts_ids)

    if shuffle:
        np.random.shuffle(contacts_ids)

    return contacts_ids

def get_sus_mort_prog_age_bracket_index(age):
        if age < 0 or age > 100:
            return None
        elif age < 10:
            return 0
        elif age < 20:
            return 1
        elif age < 30:
            return 2
        elif age < 40:
            return 3
        elif age < 50:
            return 4
        elif age < 60:
            return 5
        elif age < 70:
            return 6
        elif age < 80:
            return 7
        elif age < 90:
            return 8
        else:
            return 9
        
def get_keys_from_multidim_array(arr, idx=0):
    keys = []

    if arr is not None:
        keys = [a[idx] for a in arr]

    return keys

def is_key_in_multidim_array(arr, key, idx=0):
    return key in get_keys_from_multidim_array(arr, idx)

def get_row_from_multidim_array_by_key(arr, key, idx=0):
    keys = get_keys_from_multidim_array(arr, idx)

    key_idx = keys.index(key)

    return arr[key_idx]

def del_row_from_multidim_array_by_key(arr, key, idx=0):
    keys = get_keys_from_multidim_array(arr, idx)

    key_idx = keys.index(key)

    del arr[key_idx]

def get_all_rows_by_key(arr, key, idx=0):
    rows = []
    if arr is not None:
        rows = [a for a in arr if a[idx] == key]

    # for a in arr:
    #     if a[idx] == key:
    #         rows.append(a)

    #     if a[idx] > key:
    #         break

    return rows

def delete_all_rows_by_key(arr, key, idx=0):
    arr = [i for i in arr if i[idx] != key]

    return arr

def get_distinct_first_indices(array):
    distinct_indices = set()
    for item in array:
        distinct_indices.add(item[0])
    return list(distinct_indices)

def insert_sorted(indices, values):
    for value in values:
        bisect.insort(indices, value)

    return indices

def generate_shared_memory_int(data, type=int):
    # Create a separate Boolean array to track valid/invalid elements
    valid_mask = [x is not None for x in data]

    valid_len = sum(bool(x) for x in valid_mask)

    if valid_len == 0:
        return None
    
    # Create a shared memory block for the data array
    data_shm = shm.SharedMemory(create=True, size=valid_len * np.dtype(type).itemsize)

    # Store the data in the shared memory
    data_array = np.ndarray(valid_len, dtype=type, buffer=data_shm.buf)

    data_array_index = 0
    for i in range(len(data)):
        if valid_mask[i]:
            data_array[data_array_index] = data[i]
            data_array_index += 1

    # data_array[valid_mask] = [x for x in data if x is not None]

    # Create a shared memory block for the valid mask
    mask_shm = shm.SharedMemory(create=True, size=len(valid_mask) * np.dtype(bool).itemsize)

    # Store the valid mask in the shared memory
    mask_array = np.ndarray(len(valid_mask), dtype=bool, buffer=mask_shm.buf)
    mask_array[:] = valid_mask

    return [data_shm, mask_shm, data_array.shape, mask_array.shape]

def generate_ndarray_from_shared_memory_int(data, n_total, type=int):
    original_structured_data = []

    if data is None:
        for i in range(n_total):
            original_structured_data.append(None)
    else:    
        data_shm, mask_shm, data_shape, mask_shape = data[0], data[1], data[2], data[3]

        data_array = np.ndarray(data_shape, dtype=type, buffer=data_shm.buf)
        mask_array = np.ndarray(mask_shape, dtype=bool, buffer=mask_shm.buf)

        data_array_index = 0
        for i in range(n_total):
            if mask_array[i]:
                original_structured_data.append(data_array[data_array_index])
                data_array_index += 1
            else:
                original_structured_data.append(None)

    return original_structured_data
    
def generate_shared_memory_multidim_single(data, n_dims, dtype=int):
    if data is None:
        return None
    
    # Flatten and prepare the data
    flattened_data = []
    mask = []

    for i, sublist in enumerate(data):
        if sublist is not None:
            flattened_data.append(tuple(sublist))
            mask.append(1)           
        else:
            mask.append(0)

    total_size = len(flattened_data) * n_dims * np.dtype(dtype).itemsize
    # total_size = len(flattened_data) * np.dtype([('a', int), ('b', int)]).itemsize

    if total_size > 0:
        # Create shared memory for data
        shm_data = shm.SharedMemory(create=True, size=total_size)
        data_array = np.ndarray((len(flattened_data), n_dims), dtype=dtype, buffer=shm_data.buf)
        # data_array = np.recarray(len(flattened_data), dtype=[('a', int), ('b', int)], buf=shm_data.buf)

        # Assign values to the shared memory data array
        for i, value in enumerate(flattened_data):
            data_array[i] = value

        # Create shared memory for mask
        shm_mask = shm.SharedMemory(create=True, size=len(mask) * np.dtype(bool).itemsize)
        mask_array = np.frombuffer(shm_mask.buf, dtype=bool)

        # Assign values to the shared memory mask array
        for i, value in enumerate(mask):
            mask_array[i] = value

        # Get the names of the shared memory blocks
        return [shm_data, shm_mask, data_array.shape, mask_array.shape]
    
    return None
    
def generate_ndarray_from_shared_memory_multidim_single(data, n_total, dtype=int):
    original_structured_data = []

    if data is None:
        for i in range(n_total):
            original_structured_data.append(None)
    else: 
        data_shm, mask_shm, data_shape, mask_shape = data[0], data[1], data[2], data[3]

        data_array = np.ndarray(shape=data_shape, dtype=dtype, buffer=data_shm.buf)
        # data_array = np.recarray(shape=data_shape, dtype=[('a', int), ('b', int)])
        # data_array.data = data_shm.buf
        mask_array = np.ndarray(mask_shape, dtype=bool, buffer=mask_shm.buf)

        data_array_index = 0
        for i in range(n_total):
            if mask_array[i]:
                original_structured_data.append(data_array[data_array_index])
                data_array_index += 1
            else:
                original_structured_data.append(None)

    return original_structured_data
    
def generate_shared_memory_multidim_varying(data, n_dims, dtype=int):
    if data is None:
        return None

    # Flatten and prepare the data
    flattened_data = []
    mask = []
    indices = []

    for i, sublist in enumerate(data):
        if sublist is not None:
            mask.append(1)
            for j, item in enumerate(sublist):
                flattened_data.append(tuple(item))
                indices.append((i, j))
        else:
            mask.append(0)

    # total_size = len(flattened_data) * np.dtype([('a', int), ('b', int), ('c', int)]).itemsize
    # indices_total_size = len(indices) * np.dtype([('a', int), ('b', int)]).itemsize

    total_size = len(flattened_data) * n_dims * np.dtype(dtype).itemsize
    indices_total_size = len(indices) * 2 * np.dtype(dtype).itemsize
    
    if total_size > 0:
        # Create shared memory for data
        shm_data = shm.SharedMemory(create=True, size=total_size)

        data_array = np.ndarray((len(flattened_data), n_dims), dtype=dtype, buffer=shm_data.buf)

        # data_array = np.recarray(len(flattened_data), dtype=[('a', int), ('b', int), ('c', int)],
        #                         buf=shm_data.buf)

        # Assign values to the shared memory data array
        for i, value in enumerate(flattened_data):
            data_array[i] = value

        # Create shared memory for mask
        # shm_mask = shm.SharedMemory(create=True, size=len(mask) * np.dtype(int).itemsize)
        # mask_array = np.frombuffer(shm_mask.buf, dtype=bool)

        # # Assign values to the shared memory mask array
        # for i, value in enumerate(mask):
        #     mask_array[i] = value

        # Created shared memory for indices
        shm_indices = shm.SharedMemory(create=True, size=indices_total_size)
        indices_array = np.ndarray((len(indices), 2), dtype=int, buffer=shm_indices.buf)
        # indices_array = np.recarray(len(indices), dtype=[('a', int), ('b', int)], buf=shm_indices.buf)

        # Assign values to the shared memory mask array
        for i, value in enumerate(indices):
            indices_array[i] = value

        # Get the names of the shared memory blocks
        return [shm_data, shm_indices, data_array.shape, indices_array.shape]
    
    return None
    
def generate_ndarray_from_shared_memory_multidim_varying(data, n_total, dtype=int):
    original_structured_data = []

    if data is None:
        for i in range(n_total):
            original_structured_data.append(None)
    else:       
        data_shm, indices_shm, data_shape, indices_shape = data[0], data[1], data[2], data[3]

        data_array = np.ndarray(shape=data_shape, dtype=dtype)
        # data_array = np.recarray(shape=data_shape, dtype=[('a', int), ('b', int), ('c', int)])
        data_array.data = data_shm.buf
        indices_array = np.ndarray(indices_shape, dtype=int)
        # indices_array = np.recarray(indices_shape, dtype=[('a', int), ('b', int)])
        indices_array.data = indices_shm.buf

        original_structured_data = generate_original_structure(data_array, indices_array)

    return original_structured_data
    
def generate_original_structure(data_array, indices_array, n_total):
    original_structure = []
    
    # data_array_index = 0

    indices_rows = []

    inner_arr_index = 0

    distinct_keys = get_distinct_first_indices(indices_array)

    for i in range(n_total):
        inner_arr = []

        if len(indices_rows) == 0 and i in distinct_keys:
            indices_rows = get_all_rows_by_key(indices_array, i)

        if len(indices_rows) > 0:
            start_index = inner_arr_index + indices_rows[0][1]
            end_index = start_index + indices_rows[-1][1] + 1

            inner_arr.extend(data_array[start_index:end_index]) # non-inclusive

            inner_arr_index = end_index 

            original_structure.append(inner_arr)

            indices_rows = []
        else:
            original_structure.append(None)
    
    return original_structure