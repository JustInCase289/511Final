import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor


def matrix_multiplication(_):  # <-- Added an unused argument
    # Create two random 1,000 x 1,000 matrices
    matrix_a = np.random.rand(1000, 1000)
    matrix_b = np.random.rand(1000, 1000)

    # Multiply the matrices
    result = np.dot(matrix_a, matrix_b)

    return True


def main():
    num_of_parallel_operations = 6

    # Create the process pool outside the loop to reuse it
    with ProcessPoolExecutor(max_workers=num_of_parallel_operations) as executor:
        for _ in range(50):  # Run the main function 50 times
            start_time = time.time()

            # Execute matrix multiplications
            results = list(executor.map(matrix_multiplication, range(num_of_parallel_operations)))

            end_time = time.time()

            # Print the time taken for each run
            print(f"{end_time - start_time:.4}")


if __name__ == "__main__":
    main()
