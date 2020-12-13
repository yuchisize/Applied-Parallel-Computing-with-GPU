// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

#include <wb.h>
#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)
__global__ void total(float *input, float *output, int len) {
  //@@ Load a segment of the input vector into shared memory
  //@@ Traverse the reduction tree
  //@@ Write the computed sum of the block to the output vector at the
  //@@ correct index
  __shared__ float sdata[BLOCK_SIZE];
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  // replace the single load with two and the first step of reduction
  int index = bid*blockDim.x *2+ tid;
  float value = 0;
  value = input[index] + input[index + blockDim.x];
  sdata[tid] = value;
  __syncthreads();
  
  // //eliminate the bank conflict with reverse loop
  // for (int s = blockDim.x/2; s > 32; s>>=1){
  //     if (tid < s){
  //         sdata[tid] += sdata[tid + s];
  //     }
  //     __syncthreads();
  // }
  // unroll the loop
  if (blockDim.x >= 1024){ if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads();}
  if (blockDim.x >= 512){ if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();}
  if (blockDim.x >= 256){ if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();}
  if (blockDim.x >= 128){ if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();}
  
  // unroll the last 6 iterations
  // need __syncwarp()!!
  if (tid < 32){
      if (blockDim.x >= 64) sdata[tid] += sdata[tid + 32]; __syncwarp();
      if (blockDim.x >= 32) sdata[tid] += sdata[tid + 16]; __syncwarp();
      if (blockDim.x >= 16) sdata[tid] += sdata[tid + 8]; __syncwarp();
      if (blockDim.x >= 8) sdata[tid] += sdata[tid + 4]; __syncwarp();
      if (blockDim.x >= 4) sdata[tid] += sdata[tid + 2]; __syncwarp();
      if (blockDim.x >= 2) sdata[tid] += sdata[tid + 1]; __syncwarp();
  }
  
  if (tid == 0){
      output[bid] = sdata[0];
  }
}

int main(int argc, char **argv) {
  int ii;
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numInputElements;  // number of elements in the input list
  int numOutputElements; // number of elements in the output list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput =
      (float *)wbImport(wbArg_getInputFile(args, 0), &numInputElements);

  numOutputElements = numInputElements / (BLOCK_SIZE << 1);
  if (numInputElements % (BLOCK_SIZE << 1)) {
    numOutputElements++;
  }
  hostOutput = (float *)malloc(numOutputElements * sizeof(float));

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numInputElements);
  wbLog(TRACE, "The number of output elements in the input is ",
        numOutputElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void**) &deviceInput, numInputElements*sizeof(float));
  cudaMalloc((void**) &deviceOutput, numOutputElements*sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, numInputElements*sizeof(float), cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");
  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(numOutputElements, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  total<<<dimGrid, dimBlock>>>(deviceInput,deviceOutput,numInputElements);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, numOutputElements*sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  /********************************************************************
   * Reduce output vector on the host
   * NOTE: One could also perform the reduction of the output vector
   * recursively and support any size input. For simplicity, we do not
   * require that for this lab.
   ********************************************************************/
  for (ii = 1; ii < numOutputElements; ii++) {
    hostOutput[0] += hostOutput[ii];
  }

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, 1);

  free(hostInput);
  free(hostOutput);

  return 0;
}
