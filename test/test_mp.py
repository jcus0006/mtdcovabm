import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np

class MyClass:
    def __init__(self, shared_data):
        self.shared_data = shared_data

    def my_method(self):
        # Access shared data and perform computation
        data = np.frombuffer(self.shared_data.buf, dtype=np.int32)
        data += 1

def worker(shared_data):
    # Create an instance of MyClass using the shared data
    my_instance = MyClass(shared_data)

    # Call the method on the instance
    my_instance.my_method()

if __name__ == '__main__':
    # Create a shared memory block for data
    data_size = 10 * np.int32().itemsize
    shared_data = shared_memory.SharedMemory(create=True, size=data_size)

    # Initialize the data in the shared memory
    data = np.ndarray((10,), dtype=np.int32, buffer=shared_data.buf)
    data.fill(0)

    # Create a pool of processes
    num_processes = 4
    pool = mp.Pool(num_processes)

    # Call the worker function in parallel
    pool.map(worker, [shared_data] * num_processes)

    # Close the pool of processes
    pool.close()
    pool.join()

    # Print the modified data
    print(data)

    # Release the shared memory block
    shared_data.close()
    shared_data.unlink()


# def worker(shared_arr):
#     # Access the shared array and perform some computation
#     arr = np.frombuffer(shared_arr.buf, dtype=np.int32)
#     arr += 1

# if __name__ == '__main__':
#     # Create a shared memory block
#     size = 10 * np.int32().itemsize
#     shared_arr = mp.shared_memory.SharedMemory(create=True, size=size)

#     # Create a numpy array using the shared memory block
#     arr = np.ndarray((10,), dtype=np.int32, buffer=shared_arr.buf)

#     # Initialize the array
#     arr.fill(0)

#     # Create a pool of processes
#     num_processes = 4
#     pool = mp.Pool(num_processes)

#     # Call the worker function in parallel
#     pool.map(worker, [shared_arr] * num_processes)

#     # Close the pool of processes
#     pool.close()
#     pool.join()

#     # Print the modified array
#     print(arr)

#     # Release the shared memory block
#     shared_arr.close()
#     shared_arr.unlink()


# # In the first Python interactive shell
# import numpy as np
# from multiprocessing import shared_memory

# a = np.array([1, 1, 2, 3, 5, 8])  # Start with an existing NumPy array
# print(a)
# shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
# # Now create a NumPy array backed by shared memory
# b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
# b[:] = a[:]  # Copy the original data into shared memory

# print(b)
# print(type(b))
# print(type(a))
# print(shm.name)  # We did not specify a name so one was chosen for us

# # In either the same shell or a new Python shell on the same machine
# import numpy as np
# from multiprocessing import shared_memory
# # Attach to the existing shared memory block
# existing_shm = shared_memory.SharedMemory(name=shm.name)
# # Note that a.shape is (6,) and a.dtype is np.int64 in this example
# c = np.ndarray((6,), dtype=np.int64, buffer=existing_shm.buf)
# print(c)
# c[-1] = 888
# print(c)
# # Back in the first Python interactive shell, b reflects this change
# print(b)

# # Clean up from within the second Python shell
# del c  # Unnecessary; merely emphasizing the array is no longer used
# existing_shm.close()

# # Clean up from within the first Python shell
# del b  # Unnecessary; merely emphasizing the array is no longer used
# shm.close()
# shm.unlink()  # Free and release the shared memory block at the very end


# import multiprocessing as mp

# class Test:
#     def __init__(self, arr, dictionary, param):
#         self.arr = arr
#         self.dictionary = dictionary
#         self.param = param
    
#     def worker(self, index):
#         # Perform the work using self-contained properties and methods
#         # Modify self.arr, self.dictionary, etc. as needed
#         animal = self.arr[index]
#         return self.dictionary[animal]

# if __name__ == '__main__':
#     # Initialize your data
#     arr = ["dog", "cat", "rabbit", "hamster"]
#     dictionary = {"dog": 5, "cat": 3, "rabbit": 2, "hamster": 1}
#     param = "test"
    
#     # Create an instance of your class
#     tempTest = Test(arr, dictionary, param)
    
#     # Define the number of processes
#     num_processes = 4
    
#     # Create a pool of processes
#     pool = mp.Pool(num_processes)
    
#     # Call the worker method in parallel
#     results = pool.map(tempTest.worker, range(num_processes))
#     # results = pool.map(tempTest.worker, [(index,) for index in range(num_processes)])

#     # Close the pool of processes
#     pool.close()
#     pool.join()
    
#     # Process the results as needed
#     for result in results:
#         print("result: " + str(result))

#     print("end")


# import multiprocessing as mp
# import numpy as np

# def worker(shared_array, index):
#     # Update the shared array with the process index
#     shared_array[index] = index

# if __name__ == '__main__':
#     # Define the number of processes
#     num_processes = 4
    
#     # Create a shared array using multiprocessing.Array
#     shared_array = mp.Array('i', num_processes)
    
#     # Create a list to hold the process objects
#     processes = []
    
#     # Create and start the processes
#     for i in range(num_processes):
#         p = mp.Process(target=worker, args=(shared_array, i))
#         p.start()
#         processes.append(p)
    
#     # Wait for all processes to finish
#     for p in processes:
#         p.join()
    
#     # Print the shared array
#     print(list(shared_array))

#     print("end")