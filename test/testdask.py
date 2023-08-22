import numpy as np
import time

shared_res = None

def calculate_row_sum(result_queue, matrix, start_row, end_row, process_index, is_shared=False):
    global shared_res
    start = time.time()

    result = {}
    for i in range(10): # simulation of computation
        for row in range(start_row, end_row):
            row_sum = np.sum(matrix[row])

            if not is_shared:
                result[row] = row_sum
            else:
                shared_res[row] = row_sum

        for col in range(start_row, end_row):
            col_sum = np.sum(matrix[:, col])

            if not is_shared:
                result[col] -= col_sum
            else:
                shared_res[col] -= col_sum

    if not is_shared:
        result_queue.put((start_row, result))

    time_taken = time.time() - start
    print("process " + str(process_index) + ": " + str(time_taken))