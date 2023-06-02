import multiprocessing as mp

class Test:
    def __init__(self, arr, dictionary, param):
        self.arr = arr
        self.dictionary = dictionary
        self.param = param
    
    def worker(self, index):
        # Perform the work using self-contained properties and methods
        # Modify self.arr, self.dictionary, etc. as needed
        animal = self.arr[index]
        return self.dictionary[animal]

if __name__ == '__main__':
    # Initialize your data
    arr = ["dog", "cat", "rabbit", "hamster"]
    dictionary = {"dog": 5, "cat": 3, "rabbit": 2, "hamster": 1}
    param = "test"
    
    # Create an instance of your class
    tempTest = Test(arr, dictionary, param)
    
    # Define the number of processes
    num_processes = 4
    
    # Create a pool of processes
    pool = mp.Pool(num_processes)
    
    # Call the worker method in parallel
    results = pool.map(tempTest.worker, range(num_processes))
    # results = pool.map(tempTest.worker, [(index,) for index in range(num_processes)])

    # Close the pool of processes
    pool.close()
    pool.join()
    
    # Process the results as needed
    for result in results:
        print("result: " + str(result))

    print("end")


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