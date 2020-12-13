#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define KERNEL_SIZE 3 // mask width
#define MASK_RADIUS 1
#define TILE_SIZE KERNEL_SIZE
#define SM_SIZE (KERNEL_SIZE + (MASK_RADIUS*2))// share memory size in kernel also the block size
//@@ Define constant memory for device kernel here
__device__ __constant__ float deviceKernel[KERNEL_SIZE*KERNEL_SIZE*KERNEL_SIZE];
__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int bz = blockIdx.z*TILE_SIZE;
  int by = blockIdx.y*TILE_SIZE;
  int bx = blockIdx.x*TILE_SIZE;
  
  __shared__ float N_ds[SM_SIZE][SM_SIZE][SM_SIZE];
  // total thread
  int t = tz * (KERNEL_SIZE * KERNEL_SIZE) + ty * (KERNEL_SIZE) + tx;
  if (t < SM_SIZE * SM_SIZE){
    int input_x = bx + (t % SM_SIZE) - MASK_RADIUS;
    int input_y = by + (t / SM_SIZE) % SM_SIZE - MASK_RADIUS;
    int temp_z = bz - MASK_RADIUS;

    for (int z = 0; z < SM_SIZE; z++){
        int input_z = temp_z + z;

        if (input_x >= 0 && input_x < x_size && input_y >= 0 && input_y < y_size && input_z >= 0 && input_z < z_size){
            N_ds[t % SM_SIZE][(t / SM_SIZE) % SM_SIZE][z] = input[input_z * (y_size*x_size) + input_y * x_size + input_x];
        }
        else{
            N_ds[t % SM_SIZE][(t / SM_SIZE) % SM_SIZE][z] = 0.0f;
        }
    }
  }
  
  __syncthreads();
  float temp = 0.0f;
  // begin 3D convolution
  if ((bx + tx) >= 0 && (bx + tx) < x_size && (by + ty) >= 0 && (by + ty) < y_size && (bz + tz) >= 0 && (bz + tz) < z_size){
      for(int x = 0; x < KERNEL_SIZE; x++){
          for(int y= 0; y < KERNEL_SIZE; y++){
              for (int z = 0; z < KERNEL_SIZE; z++){
                  temp += deviceKernel[z *(KERNEL_SIZE * KERNEL_SIZE) + y * (KERNEL_SIZE) + x] * N_ds[x + tx][y + ty][z + tz];
              }
          }
      }
      output[ (bz + tz)* (y_size * x_size) + (by + ty) * (x_size) + (bx + tx)] = temp;
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  
  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void**) &deviceInput, z_size*y_size*x_size*sizeof(float));
  cudaMalloc((void**) &deviceOutput, z_size*y_size*x_size*sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");
  
  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput + 3, z_size*y_size*x_size*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(deviceKernel, hostKernel, kernelLength*sizeof(float), 0,cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 dimGrid((x_size + TILE_SIZE-1)/TILE_SIZE, (y_size + TILE_SIZE-1)/TILE_SIZE, (z_size + TILE_SIZE-1)/TILE_SIZE);
  dim3 dimBlock(TILE_SIZE, TILE_SIZE, TILE_SIZE);
  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size,y_size,x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput + 3, deviceOutput, z_size * y_size * x_size * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
