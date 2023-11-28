import threading
import numpy as np
import time
import psutil


def matrix_multiply(A, B, C, start_row, end_row):
    """Perform matrix multiplication for a specific range of rows."""
    for i in range(start_row, end_row):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i, j] += A[i, k] * B[k, j]


def threaded_matrix_multiply(A, B, num_threads=6):
    """Matrix multiplication using threading."""
    assert A.shape[1] == B.shape[0], "Matrix dimensions must be compatible for multiplication."

    # Initialize the result matrix
    C = np.zeros((A.shape[0], B.shape[1]))

    # Divide the work among threads
    threads = []
    row_step = A.shape[0] // num_threads

    for i in range(num_threads):
        start_row = i * row_step
        # Handle the last thread separately to cover any remaining rows
        end_row = (i + 1) * row_step if i != num_threads - 1 else A.shape[0]
        thread = threading.Thread(target=matrix_multiply, args=(A, B, C, start_row, end_row))
        threads.append(thread)

    # Start threads
    for thread in threads:
        thread.start()

    # Join threads
    for thread in threads:
        thread.join()

    return C


# Generate random matrices for testing
np.random.seed(0)
A = np.random.rand(100, 100)  # smaller size for testing
B = np.random.rand(100, 100)

# Run the threaded matrix multiplication 25 times and measure performance
times = []
cpu_performance_before = []
cpu_performance_after = []

for _ in range(25):
    # Record CPU performance before
    cpu_performance_before.append(psutil.cpu_percent(interval=1))

    start_time = time.time()
    C = threaded_matrix_multiply(A, B)
    end_time = time.time()

    # Record CPU performance after
    cpu_performance_after.append(psutil.cpu_percent(interval=1))

    # Store the time taken
    times.append(end_time - start_time)

# Display results
average_time = sum(times) / len(times)
average_cpu_before = sum(cpu_performance_before) / len(cpu_performance_before)
average_cpu_after = sum(cpu_performance_after) / len(cpu_performance_after)

average_time, average_cpu_before, average_cpu_after, times
