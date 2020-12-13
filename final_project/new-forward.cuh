#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#define TILE_SIZE1 16
#define TILE_SIZE2 16
#define SHARE_WIDTH1 (TILE_SIZE1+6)
#define SHARE_WIDTH2 (TILE_SIZE2+6)
namespace mxnet
{
namespace op
{
__constant__ float devicekernel[12*1*7*7];
__constant__ float devicekernel2[24*12*7*7];
__global__ void forward_kernel(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    // #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    
        int b = blockIdx.x;
        int m = blockIdx.y;
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        const int H_out = H - K + 1;
        const int W_out = W - K + 1;
        int gridlen = ceil(float(H_out)/TILE_SIZE1);
        int img_w = blockIdx.z % gridlen * TILE_SIZE1 + tx;
        int img_h = blockIdx.z / gridlen * TILE_SIZE1 + ty;
        __shared__ float X_ds[1*SHARE_WIDTH1*SHARE_WIDTH1];

        for (int c = 0; c < C; ++c){
            if (img_h < H && img_w < W){
                X_ds[c*SHARE_WIDTH1*SHARE_WIDTH1 + ty * SHARE_WIDTH1 + tx] = x4d(b,c,img_h, img_w);
            }else{
                X_ds[c*SHARE_WIDTH1*SHARE_WIDTH1 + ty * SHARE_WIDTH1 + tx] = 0.0;
            }
        }
        __syncthreads();
    
        if (img_w < 66 && img_h < 66 && tx < TILE_SIZE1 && ty < TILE_SIZE1) // for each image in the batch
        {
            y4d(b, m, img_h, img_w) = 0;
            for (int c = 0; c < C; c++)     
                for (int p = 0; p < K; p++){
                    y4d(b, m, img_h, img_w) += X_ds[c*SHARE_WIDTH1*SHARE_WIDTH1 + (ty+p) * SHARE_WIDTH1 + (tx+0)] * devicekernel[m * 49 + c * 49 + p * 7 + 0];
                    y4d(b, m, img_h, img_w) += X_ds[c*SHARE_WIDTH1*SHARE_WIDTH1 + (ty+p) * SHARE_WIDTH1 + (tx+1)] * devicekernel[m * 49 + c * 49 + p * 7 + 1];
                    y4d(b, m, img_h, img_w) += X_ds[c*SHARE_WIDTH1*SHARE_WIDTH1 + (ty+p) * SHARE_WIDTH1 + (tx+2)] * devicekernel[m * 49 + c * 49 + p * 7 + 2];
                    y4d(b, m, img_h, img_w) += X_ds[c*SHARE_WIDTH1*SHARE_WIDTH1 + (ty+p) * SHARE_WIDTH1 + (tx+3)] * devicekernel[m * 49 + c * 49 + p * 7 + 3];
                    y4d(b, m, img_h, img_w) += X_ds[c*SHARE_WIDTH1*SHARE_WIDTH1 + (ty+p) * SHARE_WIDTH1 + (tx+4)] * devicekernel[m * 49 + c * 49 + p * 7 + 4];
                    y4d(b, m, img_h, img_w) += X_ds[c*SHARE_WIDTH1*SHARE_WIDTH1 + (ty+p) * SHARE_WIDTH1 + (tx+5)] * devicekernel[m * 49 + c * 49 + p * 7 + 5];
                    y4d(b, m, img_h, img_w) += X_ds[c*SHARE_WIDTH1*SHARE_WIDTH1 + (ty+p) * SHARE_WIDTH1 + (tx+6)] * devicekernel[m * 49 + c * 49 + p * 7 + 6];
                }
            
        }
    
    #undef y4d
    #undef x4d
    // #undef k4d
}

__global__ void forward_kernel_sec(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    // #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    
        int b = blockIdx.x;
        int m = blockIdx.y;
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        const int H_out = H - K + 1;
        const int W_out = W - K + 1;
        int gridlen = ceil(float(H_out)/TILE_SIZE2);
        int img_w = blockIdx.z % gridlen * TILE_SIZE2 + threadIdx.x;
        int img_h = blockIdx.z / gridlen * TILE_SIZE2 + threadIdx.y;

        __shared__ float X_ds[12*SHARE_WIDTH2*SHARE_WIDTH2];

        for (int c = 0; c < C; ++c){
            if (img_h < H && img_w < W){
                X_ds[c*SHARE_WIDTH2*SHARE_WIDTH2 + ty * SHARE_WIDTH2 + tx] = x4d(b,c,img_h, img_w);
            }else{
                X_ds[c*SHARE_WIDTH2*SHARE_WIDTH2 + ty * SHARE_WIDTH2 + tx] = 0.0;
            }
        }
        __syncthreads();
        if (img_w < 27 && img_h < 27 && tx < TILE_SIZE2 && ty < TILE_SIZE2) // for each image in the batch
        {
            y4d(b, m, img_h, img_w) = 0;
                for (int c = 0; c < C; c++)     
                    for (int p = 0; p < K; p++){
                        y4d(b, m, img_h, img_w) += X_ds[c*SHARE_WIDTH2*SHARE_WIDTH2 + (ty+p) * SHARE_WIDTH2 + (tx+0)] * devicekernel2[m * 588 + c * 49 + p * 7 + 0];
                        y4d(b, m, img_h, img_w) += X_ds[c*SHARE_WIDTH2*SHARE_WIDTH2 + (ty+p) * SHARE_WIDTH2 + (tx+1)] * devicekernel2[m * 588 + c * 49 + p * 7 + 1];
                        y4d(b, m, img_h, img_w) += X_ds[c*SHARE_WIDTH2*SHARE_WIDTH2 + (ty+p) * SHARE_WIDTH2 + (tx+2)] * devicekernel2[m * 588 + c * 49 + p * 7 + 2];
                        y4d(b, m, img_h, img_w) += X_ds[c*SHARE_WIDTH2*SHARE_WIDTH2 + (ty+p) * SHARE_WIDTH2 + (tx+3)] * devicekernel2[m * 588 + c * 49 + p * 7 + 3];
                        y4d(b, m, img_h, img_w) += X_ds[c*SHARE_WIDTH2*SHARE_WIDTH2 + (ty+p) * SHARE_WIDTH2 + (tx+4)] * devicekernel2[m * 588 + c * 49 + p * 7 + 4];
                        y4d(b, m, img_h, img_w) += X_ds[c*SHARE_WIDTH2*SHARE_WIDTH2 + (ty+p) * SHARE_WIDTH2 + (tx+5)] * devicekernel2[m * 588 + c * 49 + p * 7 + 5];
                        y4d(b, m, img_h, img_w) += X_ds[c*SHARE_WIDTH2*SHARE_WIDTH2 + (ty+p) * SHARE_WIDTH2 + (tx+6)] * devicekernel2[m * 588 + c * 49 + p * 7 + 6];
                    }
        }
        
    #undef y4d
    #undef x4d

}


/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   We only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // // Use mxnet's CHECK_EQ to do assertions.
    // CHECK_EQ(0, 1) 
    cudaStream_t s = y.stream_->stream_;
    const int B = x.shape_[0];
    const int M = y.shape_[1]; // num_filter
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];


    if (M == 12){
        dim3 gridDim(B, M, ceil(66.0/TILE_SIZE1)*ceil(66.0/TILE_SIZE1));
        dim3 blockDim(SHARE_WIDTH1, SHARE_WIDTH1, 1);
        
        cudaMemcpyToSymbol(devicekernel, w.dptr_, 588*sizeof(float), 0, cudaMemcpyDeviceToDevice);
        MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
        forward_kernel<<<gridDim, blockDim,0,s>>>(y.dptr_, x.dptr_, B, M, C, H, W, K);
        MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    }
    else{
        dim3 gridDim(B, M, ceil(27.0/TILE_SIZE2)*ceil(27.0/TILE_SIZE2));
        dim3 blockDim(SHARE_WIDTH2, SHARE_WIDTH2, 1);

        cudaMemcpyToSymbol(devicekernel2, w.dptr_, 14112*sizeof(float), 0, cudaMemcpyDeviceToDevice);
        MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
        forward_kernel_sec<<<gridDim, blockDim,0,s>>>(y.dptr_, x.dptr_, B, M, C, H, W, K);
        MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    }
}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    assert(0 && "No forward implementation for other datatypes needed");
}
}
}

#endif
