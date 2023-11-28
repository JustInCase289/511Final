import pyopencl as cl
import pyopencl.array
import numpy as np
import time


def opencl_matrix_multiply(A, B):
    # Set up OpenCL context and queue
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    # Define the OpenCL program
    kernel_code = """
    __kernel void matmul(const int M, const int N, const int K,
                         const __global float* A,
                         const __global float* B,
                         __global float* C) {
        int row = get_global_id(1);
        int col = get_global_id(0);
        float value = 0;
        for (int e = 0; e < K; e++)
            value += A[row * K + e] * B[e * N + col];
        C[row * N + col] = value;
    }
    """
    program = cl.Program(ctx, kernel_code).build()

    # Create buffers
    A_buf = cl.array.to_device(queue, A)
    B_buf = cl.array.to_device(queue, B)
    C_buf = cl.array.empty(queue, A.shape, np.float32)

    # Execute OpenCL program
    program.matmul(queue, A.shape, None,
                   np.int32(A.shape[0]), np.int32(A.shape[1]), np.int32(B.shape[1]),
                   A_buf.data, B_buf.data, C_buf.data)

    return C_buf.get()


def main():
    A = np.random.rand(1000, 1000).astype(np.float32)
    B = np.random.rand(1000, 1000).astype(np.float32)

    start_time = time.time()
    result = opencl_matrix_multiply(A, B)
    end_time = time.time()

    print(f"{end_time - start_time:.4f}")


if __name__ == "__main__":
    for _ in range(50):
        main()
