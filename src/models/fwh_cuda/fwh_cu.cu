// The codes are from Armen Aghajanyan from facebook, from paper
// Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning
// https://arxiv.org/abs/2012.13255

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ void FastWalshHadamardKernel(const int stride, const scalar_t* in, scalar_t* out) {
    const auto idx = (threadIdx.x + blockIdx.x * blockDim.x);
    const auto elemIdx = (idx / stride ) * (2 * stride) + (idx % stride);
    const auto tmp = in[elemIdx], tmp2 = in[elemIdx + stride];
    out[elemIdx] = tmp + tmp2;
    out[elemIdx + stride] = tmp - tmp2;
}

template <typename scalar_t>
__global__ void FastWalshHadamardSubKernel(const scalar_t scalar, scalar_t* out) {
    const auto idx = (threadIdx.x + blockIdx.x * blockDim.x);
    out[idx] *= scalar;
}


void fast_walsh_hadamard_transform_cuda_kernel(const int NN, const int halfLL, torch::Tensor in, torch::Tensor out, bool normalize) {
    // Apply Unnormalized Fast Walsh Hadamard transform
    int stride = halfLL;
    float normalizer = 1.0;
    float sqrt2inv = 0.70710678118654746;
    
    while (stride >= 1) {
      if(stride == halfLL)
      {
        AT_DISPATCH_FLOATING_TYPES(in.scalar_type(),"fast_walsh_hadamard_transform_in", ([&] {
            FastWalshHadamardKernel<scalar_t><<<max(1, halfLL/256), min(256, halfLL)>>>(stride, in.data_ptr<scalar_t>(), out.data_ptr<scalar_t>());
          }));
      }
      else
      {
        AT_DISPATCH_FLOATING_TYPES(in.scalar_type(),"fast_walsh_hadamard_transform_out", ([&] {
            FastWalshHadamardKernel<scalar_t><<<max(1, halfLL/256), min(256, halfLL)>>>(stride, out.data_ptr<scalar_t>(), out.data_ptr<scalar_t>());
          }));
      }

      stride /= 2;
      normalizer *= sqrt2inv;
    }
    if(normalize){
      AT_DISPATCH_FLOATING_TYPES(in.scalar_type(),"fast_walsh_hadamard_transform_final", ([&] {
        FastWalshHadamardSubKernel<scalar_t><<<max(1, NN/256), min(256, NN)>>>(normalizer, out.data_ptr<scalar_t>());
      }));
    }
}