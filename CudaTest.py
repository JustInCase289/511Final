import cupy as cp
import time


def gpu_matrix_multiplication(_):
    # Create two random 1,000 x 1,000 matrices on the GPU
    matrix_a = cp.random.rand(1000, 1000)
    matrix_b = cp.random.rand(1000, 1000)

    # Multiply the matrices on the GPU
    result = cp.dot(matrix_a, matrix_b)

    return True


def main():
    num_of_parallel_operations = 6  # This won't have the same meaning as CPU parallelism
    start_time = time.time()

    # CuPy will handle the GPU parallelism internally
    for _ in range(num_of_parallel_operations):
        gpu_matrix_multiplication(None)

    end_time = time.time()

    print(f"{end_time - start_time:.4f}")


if __name__ == "__main__":
    for _ in range(50):
        main()
