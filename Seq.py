import numpy as np
import time


def matrix_multiplication():
    # Create two random 1,000 x 1,000 matrices
    matrix_a = np.random.rand(1000, 1000)
    matrix_b = np.random.rand(1000, 1000)

    # Multiply the matrices
    result = np.dot(matrix_a, matrix_b)

    return True


def main():
    num_of_operations_per_loop = 6
    total_loops = 50

    for i in range(total_loops):
        start_time = time.time()

        # Perform matrix multiplications 6 times per loop
        for _ in range(num_of_operations_per_loop):
            matrix_multiplication()

        end_time = time.time()

        print(f"{end_time - start_time:.4}")


if __name__ == "__main__":
    main()
