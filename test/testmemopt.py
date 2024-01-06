import sys
import time
from memory_profiler import profile
from pympler import asizeof
import numpy as np
import cloudpickle
from enum import Enum

class Property(Enum):
    IT = 0
    ITND = 1
    NDAR = 2
    PDNDAR = 3

# class AgentsUtil:
#     def __init__(self, agents):
#         self.agents = agents

#     def get(self, prop_index, prop_name):
#         match prop_name:
#             case "itinerary":
#                 return self.agents[prop_index][Property.IT.value]
#             case "itinerary_nextday":
#                 return self.agents[prop_index][Property.ITND.value]
#             case "non_daily_activity_recurring":
#                 return self.agents[prop_index][Property.NDAR.value]
#             case "prevday_non_daily_activity_recurring":
#                 return self.agents[prop_index][Property.PDNDAR.value]
            
#     def set(self, prop_index, prop_name, prop_value):
#         match prop_name:
#             case "itinerary":
#                 self.agents[prop_index][Property.IT.value] = prop_value
#             case "itinerary_nextday":
#                 self.agents[prop_index][Property.ITND.value] = prop_value
#             case "non_daily_activity_recurring":
#                 self.agents[prop_index][Property.NDAR.value] = prop_value
#             case "prevday_non_daily_activity_recurring":
#                 self.agents[prop_index][Property.PDNDAR.value] = prop_value

prop_indices = {"itinerary": 0, "itinerary_nextday": 1, "non_daily_activity_recurring": 2, "prevday_non_daily_activity_recurring": 3}
prop_indices_arr = ["itinerary", "itinerary_nextday", "non_daily_activity_recurring", "prevday_non_daily_activity_recurring"]

def get_property_index(prop_name):
    return prop_indices[prop_name]
    # return prop_indices_arr.index(prop_name)

# fp = open("testmemopt_profiler_dictvsarr.log", "w+")
# @profile(stream=fp)
def main():

    # demonstration of string interning (all references to key1 will refer to the same string in memory)
    a = "key1"
    b = "key1"
    x = {1: {"key1": "val1", "key2": "val2"},
         2: {"key1": "val3", "key2": "val4"},
         3: {"key1": "val5", "key2": "val6"}}
    
    y = {1: ["val1", "val2"],
         2: ["val3", "val4"],
         3: ["val5", "val6"]}
    
    x_bytes = cloudpickle.dumps(x)
    y_bytes = cloudpickle.dumps(y)

    x_b_size = sys.getsizeof(x_bytes)
    y_b_size = sys.getsizeof(y_bytes)

    print(f"x bytes size {x_b_size}")
    print(f"y bytes size {y_b_size}")

    print(f"a_id: {id(a)}")
    print(f"b_id: {id(b)}")

    print([id(j) for i in iter(x) for j in iter(x[i])])

    x[4] = {"key1": "val7", "key2": "val8"}
    x[5] = {"key1": "val9", "key2": "val10"}
    x[6] = {"key1": "val11", "key2": "val12"}

    print([id(j) for i in iter(x) for j in iter(x[i])])

    # test with multi dimensional array vs dict - the keys will utilise more data (58% more)
    # arraytest = [["abc", "xyz", "def", "klm"] for i in range(10000000)]
    # dicttest = {i: ["abc", "xyz", "def", "klm"] for i in range(10000000)}

    # print(f"arraytest {round(asizeof.asizeof(arraytest)/ (1024 * 1024), 2)}")
    # print(f"dicttest {round(asizeof.asizeof(dicttest)/ (1024 * 1024), 2)}")

    # range_nums = np.arange(1_000_000)
    # random_nums = np.random.choice(range_nums, size=50000)
    # print("testing the array")
    # start = time.time()
    # for i in range(50000):   
    #     x = arraytest[random_nums[i]]
    #     # print(x)

    # time_taken = time.time() - start
    # print(f"read 1000 random indices {time_taken}")

    # print("testing the dict")
    # start = time.time()
    # for i in range(50000):
    #     x = dicttest[random_nums[i]]
    #     # print(x)

    # time_taken = time.time() - start
    # print(f"read 1000 random indices {time_taken}")
    
    # print("deleting structs")
    # arraytest = []
    # dicttest = {}
    
    # print(f"arraytest {round(asizeof.asizeof(arraytest)/ (1024 * 1024), 2)}")
    # print(f"dicttest {round(asizeof.asizeof(dicttest)/ (1024 * 1024), 2)}")
    
    # testing with dict with inner dicts vs inner arrays (inner dict uses 23% more memory, array reads over 500000 reads are 4% slower)
    # but this is not because of the extra keys. probably in general dicts use more memory because of the hash map that they need to maintain
    # the extra keys are actually using a memory reference to the same immutable string in memory 
    # however, the problem lies when pickling. the dict is not optimised for pickling
    # first 
    print("initializing inner dicts as {}")
    dict_arr = {i:[{}, {}, {}, {}] for i in range(500000)}
    dict_dict = {i:{"itinerary": {}, "itinerary_nextday": {}, "non_daily_activity_recurring":{}, "prevday_non_daily_activity_recurring": {}} for i in range(500000)}
    
    print("showing memory ids of first 10 agent ids")
    print([(i, j, id(j)) for i in iter(dict_dict) if i < 10 for j in iter(dict_dict[i])])

    dict_arr_bytes = cloudpickle.dumps(dict_arr)
    dict_dict_bytes = cloudpickle.dumps(dict_dict)

    dict_arr_size = round(sys.getsizeof(dict_arr_bytes) / (1024 * 1024), 2)
    dict_dict_size = round(sys.getsizeof(dict_dict_bytes) / (1024 * 1024), 2)

    print(f"dict_arr bytes (empty) size {dict_arr_size}")
    print(f"dict_dict bytes (empty) size {dict_dict_size}")
    
    dict_arr_size = asizeof.asizeof(dict_arr)
    dict_dict_size = asizeof.asizeof(dict_dict)
    diff_size = dict_dict_size - dict_arr_size
    print(f"dict_arr size (empty): {round(dict_arr_size / (1024 * 1024), 2)} mb")
    print(f"dict_dict size (empty): {round(dict_dict_size / (1024 * 1024), 2)} mb")
    print(f"diff size (empty): {round(diff_size / (1024 * 1024), 2)} mb")

    print("initializing inner dicts as None")
    dict_arr = {i:[None, None, None, None, None] for i in range(500000)}
    dict_dict = {i:{"itinerary": None, "itinerary_nextday": None, "non_daily_activity_recurring":None, "prevday_non_daily_activity_recurring": None} for i in range(500000)}
    
    print("showing memory ids of first 10 agent ids")
    print([(i, j, id(j)) for i in iter(dict_dict) if i < 10 for j in iter(dict_dict[i])])

    dict_arr_bytes = cloudpickle.dumps(dict_arr)
    dict_dict_bytes = cloudpickle.dumps(dict_dict)

    dict_arr_size = round(sys.getsizeof(dict_arr_bytes) / (1024 * 1024), 2)
    dict_dict_size = round(sys.getsizeof(dict_dict_bytes) / (1024 * 1024), 2)

    print(f"dict_arr bytes (empty) size {dict_arr_size}")
    print(f"dict_dict bytes (empty) size {dict_dict_size}")
    
    dict_arr_size = asizeof.asizeof(dict_arr)
    dict_dict_size = asizeof.asizeof(dict_dict)
    diff_size = dict_dict_size - dict_arr_size
    print(f"dict_arr size (empty): {round(dict_arr_size / (1024 * 1024), 2)} mb")
    print(f"dict_dict size (empty): {round(dict_dict_size / (1024 * 1024), 2)} mb")
    print(f"diff size (empty): {round(diff_size / (1024 * 1024), 2)} mb")

    # dict_arr_write_sum = 0
    # dict_dict_write_sum = 0

    # agents_util = AgentsUtil(dict_arr)
    start = time.time()
    for i in range(500000):
        this_agent = dict_arr[i]
        this_agent[Property.IT.value] = {"key1": i*1, "key2": i*2, "key3": i*3, "key4": i*4}
        this_agent[Property.ITND.value] = {"key5": i*1, "key6": i*2, "key7": i*3, "key8": i*4}
        this_agent[Property.NDAR.value] = {"key9": i*1, "key10": i*2, "key11": i*3, "key12": i*4}
        this_agent[Property.PDNDAR.value] = {"key13": i*1, "key14": i*2, "key15": i*3, "key16": i*4}
        # this_agent[get_property_index("itinerary")] = {"key1": i*1, "key2": i*2, "key3": i*3, "key4": i*4}
        # this_agent[get_property_index("itinerary_nextday")] = {"key5": i*1, "key6": i*2, "key7": i*3, "key8": i*4}
        # this_agent[get_property_index("non_daily_activity_recurring")] = {"key9": i*1, "key10": i*2, "key11": i*3, "key12": i*4}
        # this_agent[get_property_index("prevday_non_daily_activity_recurring")] = {"key13": i*1, "key14": i*2, "key15": i*3, "key16": i*4}
        # this_agent[0] = {"key1": i*1, "key2": i*2, "key3": i*3, "key4": i*4}
        # this_agent[1] = {"key5": i*1, "key6": i*2, "key7": i*3, "key8": i*4}
        # this_agent[2] = {"key9": i*1, "key10": i*2, "key11": i*3, "key12": i*4}
        # this_agent[3] = {"key13": i*1, "key14": i*2, "key15": i*3, "key16": i*4}
    time_taken = time.time() - start
    print(f"dict_arr time taken to write 500000 x 4 values {time_taken}")

    start = time.time()
    for i in range(500000):
        this_agent = dict_dict[i]
        this_agent["itinerary"] = {"key1": i*1, "key2": i*2, "key3": i*3, "key4": i*4}
        this_agent["itinerary_nextday"] = {"key5": i*1, "key6": i*2, "key7": i*3, "key8": i*4}
        this_agent["non_daily_activity_recurring"] = {"key9": i*1, "key10": i*2, "key11": i*3, "key12": i*4}
        this_agent["prevday_non_daily_activity_recurring"] = {"key13": i*1, "key14": i*2, "key15": i*3, "key16": i*4}
    time_taken = time.time() - start
    print(f"dict_dict time taken to write 500000 x 4 values {time_taken}")

    # for i in range(500000):
    #     arr_start = time.time()
    #     # agents_util.set(i, "itinerary", {"key1": i*1, "key2": i*2, "key3": i*3, "key4": i*4})
    #     # agents_util.set(i, "itinerary_nextday", {"key5": i*1, "key6": i*2, "key7": i*3, "key8": i*4})
    #     # agents_util.set(i, "non_daily_activity_recurring", {"key9": i*1, "key10": i*2, "key11": i*3, "key12": i*4})
    #     # agents_util.set(i, "prevday_non_daily_activity_recurring", {"key13": i*1, "key14": i*2, "key15": i*3, "key16": i*4})

    #     # dict_arr[i][Property.IT.value] = {"key1": i*1, "key2": i*2, "key3": i*3, "key4": i*4}
    #     # dict_arr[i][Property.ITND.value] = {"key5": i*1, "key6": i*2, "key7": i*3, "key8": i*4}
    #     # dict_arr[i][Property.NDAR.value] = {"key9": i*1, "key10": i*2, "key11": i*3, "key12": i*4}
    #     # dict_arr[i][Property.PDNDAR.value] = {"key13": i*1, "key14": i*2, "key15": i*3, "key16": i*4}

    #     dict_arr[i][0] = {"key1": i*1, "key2": i*2, "key3": i*3, "key4": i*4}
    #     dict_arr[i][1] = {"key5": i*1, "key6": i*2, "key7": i*3, "key8": i*4}
    #     dict_arr[i][2] = {"key9": i*1, "key10": i*2, "key11": i*3, "key12": i*4}
    #     dict_arr[i][3] = {"key13": i*1, "key14": i*2, "key15": i*3, "key16": i*4}

    #     # dict_arr[i][get_property_index("itinerary")] = {"key1": i*1, "key2": i*2, "key3": i*3, "key4": i*4}
    #     # dict_arr[i][get_property_index("itinerary_nextday")] = {"key5": i*1, "key6": i*2, "key7": i*3, "key8": i*4}
    #     # dict_arr[i][get_property_index("non_daily_activity_recurring")] = {"key9": i*1, "key10": i*2, "key11": i*3, "key12": i*4}
    #     # dict_arr[i][get_property_index("prevday_non_daily_activity_recurring")] = {"key13": i*1, "key14": i*2, "key15": i*3, "key16": i*4}

    #     arr_time_taken = time.time() - arr_start
    #     dict_arr_write_sum += arr_time_taken

    #     dict_start = time.time()
    #     this_agent = dict_dict[i]
    #     this_agent["itinerary"] = {"key1": i*1, "key2": i*2, "key3": i*3, "key4": i*4}
    #     this_agent["itinerary_nextday"] = {"key5": i*1, "key6": i*2, "key7": i*3, "key8": i*4}
    #     this_agent["non_daily_activity_recurring"] = {"key9": i*1, "key10": i*2, "key11": i*3, "key12": i*4}
    #     this_agent["prevday_non_daily_activity_recurring"] = {"key13": i*1, "key14": i*2, "key15": i*3, "key16": i*4}
    #     dict_time_taken = time.time() - dict_start
    #     dict_dict_write_sum += dict_time_taken

    dict_arr_bytes = cloudpickle.dumps(dict_arr)
    dict_dict_bytes = cloudpickle.dumps(dict_dict)

    dict_arr_size = round(sys.getsizeof(dict_arr_bytes) / (1024 * 1024), 2)
    dict_dict_size = round(sys.getsizeof(dict_dict_bytes) / (1024 * 1024), 2)

    print(f"dict_arr bytes size {dict_arr_size}")
    print(f"dict_dict bytes size {dict_dict_size}")

    dict_arr_size = asizeof.asizeof(dict_arr)
    dict_dict_size = asizeof.asizeof(dict_dict)
    diff_size = dict_dict_size - dict_arr_size
    print(f"dict_arr size: {round(dict_arr_size / (1024 * 1024), 2)} mb")
    print(f"dict_dict size: {round(dict_dict_size / (1024 * 1024), 2)} mb")
    print(f"diff size: {round(diff_size / (1024 * 1024), 2)} mb")

    start = time.time()
    for i in range(500000):
        # x = agents_util.get(i, "itinerary")
        # x = agents_util.get(i, "itinerary_nextday")
        # x = agents_util.get(i, "non_daily_activity_recurring")
        # x = agents_util.get(i, "prevday_non_daily_activity_recurring")

        this_agent = dict_arr[i]
        x = this_agent[Property.IT.value]
        x = this_agent[Property.ITND.value]
        x = this_agent[Property.NDAR.value]
        x = this_agent[Property.PDNDAR.value]
        # x = this_agent[0]
        # x = this_agent[1]
        # x = this_agent[2]
        # x = this_agent[3]
        # x = this_agent[get_property_index("itinerary")]
        # x = this_agent[get_property_index("itinerary_nextday")]
        # x = this_agent[get_property_index("non_daily_activity_recurring")]
        # x = this_agent[get_property_index("prevday_non_daily_activity_recurring")]

    time_taken = time.time() - start
    print(f"arr mode 500000 reads time_taken {time_taken}")

    start = time.time()
    for i in range(500000):
        this_agent = dict_dict[i]
        x = this_agent["itinerary"]
        x = this_agent["itinerary_nextday"]
        x = this_agent["non_daily_activity_recurring"]
        x = this_agent["prevday_non_daily_activity_recurring"]
    time_taken = time.time() - start
    print(f"dict mode 500000 reads time_taken {time_taken}")

if __name__ == '__main__':
    main()
