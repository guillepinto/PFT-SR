#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <vector>


///////////// SMM_QmK

// CUDA kernel for forward propagation
__global__ void SMM_QmK_forward_kernel(const float* A, const float* B, const int* index, float* C, int Batch, int N, int K, int C_dim, int B_cols) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Corresponds to N
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Corresponds to K

    if (row < N && col < K) {
        int b_col = index[batch * N * K + row * K + col];
        float value = 0.0;
        for (int e = 0; e < C_dim; ++e) {
            // A (Batch, N, C) and B (Batch, C, N)
            value += A[batch * N * C_dim + row * C_dim + e] * B[batch * C_dim * B_cols + e * B_cols + b_col];
        }
        C[batch * N * K + row * K + col] = value;
    }
}

// Forward propagation function
at::Tensor SMM_QmK_forward_cuda(const at::Tensor &A, const at::Tensor &B, const at::Tensor &index) {

    // Check if tensors are contiguous
    AT_ASSERTM(A.is_contiguous(), "A tensor must be contiguous");
    AT_ASSERTM(B.is_contiguous(), "B tensor must be contiguous");
    AT_ASSERTM(index.is_contiguous(), "Index tensor must be contiguous");

    const int Batch = A.size(0);
    const int N = A.size(1);   // Dimension N of A
    const int C_dim = A.size(2);  // Dimension C of A (which is the row count of B)
    const int K = index.size(2);
    const int B_cols = B.size(2);  // Column count of B

    auto C = at::zeros({Batch, N, K}, A.options().dtype(torch::kFloat32));

    const int threads = 32;
    const dim3 block_dim(threads, threads);
    const dim3 grid_dim((K + threads - 1) / threads, (N + threads - 1) / threads, Batch);

    SMM_QmK_forward_kernel<<<grid_dim, block_dim>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), index.data_ptr<int>(), C.data_ptr<float>(), Batch, N, K, C_dim, B_cols
    );

    return C;
}

// CUDA kernel for backward propagation
__global__ void SMM_QmK_backward_kernel(const float* grad_output, const float* A, const float* B, const int* index, float* grad_A, float* grad_B, int Batch, int N, int K, int C_dim, int B_cols) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < K) {
        int b_col = index[batch * N * K + row * K + col];

        // Compute the index of grad_output and get its gradient value
        int grad_output_idx = batch * N * K + row * K + col;
        float grad_value = grad_output[grad_output_idx];

        for (int e = 0; e < C_dim; ++e) {
            // Accumulate gradients into grad_A and grad_B
            atomicAdd(&grad_A[batch * N * C_dim + row * C_dim + e], grad_value * B[batch * C_dim * B_cols + e * B_cols + b_col]);
            atomicAdd(&grad_B[batch * C_dim * B_cols + e * B_cols + b_col], grad_value * A[batch * N * C_dim + row * C_dim + e]);
        }
    }
}

std::vector<at::Tensor> SMM_QmK_backward_cuda(const at::Tensor &grad_output,
                                                       const at::Tensor &A,
                                                       const at::Tensor &B,
                                                       const at::Tensor &index) {
    // Check the contiguity and device of the inputs
    AT_ASSERTM(A.is_contiguous(), "A tensor has to be contiguous");
    AT_ASSERTM(B.is_contiguous(), "B tensor has to be contiguous");
    AT_ASSERTM(index.is_contiguous(), "index tensor has to be contiguous");
    AT_ASSERTM(grad_output.is_contiguous(), "grad_output tensor has to be contiguous");

    // Get dimensions of A and B
    const int Batch = A.size(0);
    const int N = A.size(1);    // Corresponds to dimension N of A
    const int C_dim = A.size(2); // Corresponds to dimension C of A
    const int K = index.size(2); // Corresponds to dimension K of index
    const int B_cols = B.size(2); // Corresponds to the column count of B

    // Allocate gradient tensors
    auto grad_A = at::zeros_like(A);
    auto grad_B = at::zeros_like(B);

    // Define the size of the CUDA blocks and grids
    const int threads = 32;
    const dim3 block_dim(threads, threads);
    const dim3 grid_dim((K + threads - 1) / threads, (N + threads - 1) / threads, Batch);

    // Launch the CUDA kernel to perform the backward operation
    SMM_QmK_backward_kernel<<<grid_dim, block_dim>>>(
        grad_output.data_ptr<float>(),
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        index.data_ptr<int>(),
        grad_A.data_ptr<float>(),
        grad_B.data_ptr<float>(),
        Batch,
        N,
        K,
        C_dim,
        B_cols
    );

    return {grad_A, grad_B};
}





///////////// SMM_AmV

// CUDA kernel for forward propagation
__global__ void SMM_AmV_forward_kernel(const float* A, const float* B, const int* index, float* C, int Batch, int M, int K, int B_cols) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < B_cols) {
        float value = 0.0;
        for (int e = 0; e < K; ++e) {
            int b_row = index[batch * M * K + row * K + e];
            value += A[batch * M * K + row * K + e] * B[batch * M * B_cols + b_row * B_cols + col];
        }
        C[batch * M * B_cols + row * B_cols + col] = value;
    }
}


// Forward propagation function
at::Tensor SMM_AmV_forward_cuda(const at::Tensor &A, const at::Tensor &B, const at::Tensor &index) {
    // Ensure the tensors are contiguous and on the correct device
    AT_ASSERTM(A.is_contiguous(), "A tensor must be contiguous");
    AT_ASSERTM(B.is_contiguous(), "B tensor must be contiguous");
    AT_ASSERTM(index.is_contiguous(), "Index tensor must be contiguous");

    const int Batch = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int B_cols = B.size(2);

    auto C = at::zeros({Batch, M, B_cols}, A.options().dtype(torch::kFloat32));

    const int threads = 32;
    const dim3 block_dim(threads, threads);
    const dim3 grid_dim((B_cols + threads - 1) / threads, (M + threads - 1) / threads, Batch);

    SMM_AmV_forward_kernel<<<grid_dim, block_dim>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), index.data_ptr<int>(), C.data_ptr<float>(), Batch, M, K, B_cols
    );

    return C;
}


// CUDA kernel for backward propagation
__global__ void SMM_AmV_backward_kernel(const float* grad_output, const float* A, const float* B, const int* index, float* grad_A, float* grad_B, int Batch, int M, int K, int B_cols) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < B_cols) {
        float grad_value = grad_output[batch * M * B_cols + row * B_cols + col];
        for (int e = 0; e < K; ++e) {
            int b_row = index[batch * M * K + row * K + e];
            atomicAdd(&grad_A[batch * M * K + row * K + e], grad_value * B[batch * M * B_cols + b_row * B_cols + col]);
            atomicAdd(&grad_B[batch * M * B_cols + b_row * B_cols + col], grad_value * A[batch * M * K + row * K + e]);
        }
    }
}


// Backward propagation function
std::vector<at::Tensor> SMM_AmV_backward_cuda(const at::Tensor &grad_output, const at::Tensor &A, const at::Tensor &B, const at::Tensor &index) {
    // Ensure tensors are contiguous and on the correct device
    AT_ASSERTM(A.is_contiguous(), "A tensor has to be contiguous");
    AT_ASSERTM(B.is_contiguous(), "B tensor has to be contiguous");
    AT_ASSERTM(index.is_contiguous(), "Index tensor has to be contiguous");
    AT_ASSERTM(grad_output.is_contiguous(), "grad_output tensor has to be contiguous");

    const int Batch = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int B_cols = B.size(2);

    auto grad_A = at::zeros_like(A);
    auto grad_B = at::zeros_like(B);

    const int threads = 32;
    const dim3 block_dim(threads, threads);
    const dim3 grid_dim((B_cols + threads - 1) / threads, (M + threads - 1) / threads, Batch);

    SMM_AmV_backward_kernel<<<grid_dim, block_dim>>>(
        grad_output.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), index.data_ptr<int>(), grad_A.data_ptr<float>(), grad_B.data_ptr<float>(), Batch, M, K, B_cols
    );

    return {grad_A, grad_B};
}




// Module registration
PYBIND11_MODULE(smm_cuda, m) {
    m.def("SMM_QmK_forward_cuda", &SMM_QmK_forward_cuda, "Sparse Matrix Multiplication Forward for Q @ K (CUDA)");
    m.def("SMM_QmK_backward_cuda", &SMM_QmK_backward_cuda, "Sparse Matrix Multiplication Backward for Q @ K (CUDA)");

    m.def("SMM_AmV_forward_cuda", &SMM_AmV_forward_cuda, "Sparse Matrix Multiplication Forward for A @ V  (CUDA)");
    m.def("SMM_AmV_backward_cuda", &SMM_AmV_backward_cuda, "Sparse Matrix Multiplication Backward for A @ V(CUDA)");
}
