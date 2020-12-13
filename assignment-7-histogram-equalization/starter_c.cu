// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 32
//@@ insert code here
//@@ =======================KERNEL=======================
//@@ Cast the image from float to unsigned char
__global__ void floatToChar(float *input, unsigned char *output, int width, int height){
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x*blockDim.x + tx;
  int y = blockIdx.y*blockDim.y + ty;
  int index = blockIdx.z * (width * height) + y * width + x; 

  if (x < width && y < height){
    output[index] = (unsigned char) (255*input[index]);
  }
}
//@@ Convert the image from RGB to GrayScale
__global__ void rgbToGray(unsigned char *input, unsigned char *output, int width, int height){
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x*blockDim.x + tx;
  int y = blockIdx.y*blockDim.y + ty;
  int index = y*(width) + x;

  if (x < width && y < height){
    unsigned char r = input[3*index];
    unsigned char g = input[3*index + 1];
    unsigned char b = input[3*index + 2];
    output[index] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
  }
}
//@@ Compute the histogram of grayImage
__global__ void computeHistogram(unsigned char *input, unsigned int *output, int width, int height){
  __shared__ unsigned int hist[HISTOGRAM_LENGTH];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = ty*blockDim.x + tx;
  int x = blockIdx.x*blockDim.x + tx;
  int y = blockIdx.y*blockDim.y + ty;
  int index = y*(width) + x;

  if (tid < HISTOGRAM_LENGTH){
    hist[tid] = 0;
  }
  __syncthreads();
  if (x < width && y < height){
    unsigned char data = input[index];
    atomicAdd(&(hist[data]), 1);
  }
  __syncthreads();
  if (tid < HISTOGRAM_LENGTH){
    atomicAdd(&(output[tid]), hist[tid]);
  }

}
//@@ Compute the cumulative distribution function
__global__ void computeCDF(unsigned int *input, float *output, int width, int height){
   __shared__ unsigned int cdf[HISTOGRAM_LENGTH];
   int tx = threadIdx.x;
   cdf[tx] = input[tx];
   // first scan
   for (int stride = 1; stride <= HISTOGRAM_LENGTH/2; stride<<=1){
     // need syncthread before doing the scan
    __syncthreads();
     int index = (tx+1)*stride*2-1;
     if (index<HISTOGRAM_LENGTH){
       cdf[index] += cdf[index-stride];
     }
   }
   // post scan
   for (int stride = HISTOGRAM_LENGTH/4; stride > 0; stride>>=1){
    __syncthreads();
    int index = (tx+1)*stride*2-1;
    if (index+stride<HISTOGRAM_LENGTH){
      cdf[index+stride] += cdf[index];
    }
   }
   __syncthreads();
   output[tx] = cdf[tx]/((float)(width*height));

}
//@@ Define the histogram equalization function
//@@ Apply the histogram equalization function
__global__ void histEqualize(float *cdf, unsigned char *output, int width, int height){
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x*blockDim.x + tx;
  int y = blockIdx.y*blockDim.y + ty;
  int index = blockIdx.z * (width * height) + y * width + x; 

  if (x < width && y < height){
    unsigned char temp = output[index];
    float equalized = 255 * (cdf[temp] - cdf[0]/(1.0 - cdf[0]));
    float corrected = min(max(equalized, 0.0), 255.0);

    output[index] = (unsigned char) (corrected); 
  }
}

//@@ Cast back to float
__global__ void charToFloat(unsigned char *input, float *output, int width, int height){
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x*blockDim.x + tx;
  int y = blockIdx.y*blockDim.y + ty;
  int index = blockIdx.z * (width * height) + y * (width) + x;

  if (x < width && y < height){
    output[index] = (float) (input[index]/255.0);
  }

}
//@@ =======================CPU=======================
int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceImageFloat;
  unsigned char *deviceImageChar;
  unsigned char *deviceImageCharGray;
  unsigned int *deviceImageHistogram;
  float *deviceImageCDF;
  
  float *inputImageData;
  float *outputImageData;
  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  inputImageData = wbImage_getData(inputImage);
  outputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void**) &deviceImageFloat, imageWidth*imageHeight*imageChannels*sizeof(float));
  cudaMalloc((void**) &deviceImageChar, imageWidth*imageHeight*imageChannels*sizeof(unsigned char));
  cudaMalloc((void**) &deviceImageCharGray, imageWidth*imageHeight*sizeof(unsigned char));
  cudaMalloc((void**) &deviceImageHistogram, HISTOGRAM_LENGTH*sizeof(unsigned int));
  cudaMemset((void*) &deviceImageHistogram, 0, HISTOGRAM_LENGTH*sizeof(unsigned int));
  cudaMalloc((void**) &deviceImageCDF, HISTOGRAM_LENGTH*sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory.");
  
  cudaMemcpy(deviceImageFloat, inputImageData, imageWidth*imageHeight*imageChannels*sizeof(float), cudaMemcpyHostToDevice);
  dim3 dimGrid;
  dim3 dimBlock;
  dimGrid = dim3((imageWidth+BLOCK_SIZE-1)/BLOCK_SIZE, (imageHeight+BLOCK_SIZE-1)/BLOCK_SIZE, imageChannels);
  dimBlock = dim3(BLOCK_SIZE,BLOCK_SIZE,1);
  
  // float to char
  floatToChar<<<dimGrid, dimBlock>>>(deviceImageFloat, deviceImageChar, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  
  // RGB to Grey scale
  dimGrid = dim3((imageWidth+BLOCK_SIZE-1)/BLOCK_SIZE, (imageHeight+BLOCK_SIZE-1)/BLOCK_SIZE, 1);
  dimBlock = dim3(BLOCK_SIZE,BLOCK_SIZE,1);
  rgbToGray<<<dimGrid, dimBlock>>>(deviceImageChar, deviceImageCharGray, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  
  // Compute Histogram
  dimGrid = dim3((imageWidth+BLOCK_SIZE-1)/BLOCK_SIZE, (imageHeight+BLOCK_SIZE-1)/BLOCK_SIZE, 1);
  dimBlock = dim3(BLOCK_SIZE,BLOCK_SIZE,1);
  computeHistogram<<<dimGrid, dimBlock>>>(deviceImageCharGray, deviceImageHistogram, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  // Compute CDF
  dimGrid = dim3(1, 1, 1);
  dimBlock = dim3(HISTOGRAM_LENGTH, 1, 1);
  computeCDF<<<dimGrid, dimBlock>>>(deviceImageHistogram, deviceImageCDF, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  //histogram equalize function
  dimGrid = dim3((imageWidth+BLOCK_SIZE-1)/BLOCK_SIZE, (imageHeight+BLOCK_SIZE-1)/BLOCK_SIZE, imageChannels);
  dimBlock = dim3(BLOCK_SIZE,BLOCK_SIZE,1);
  histEqualize<<<dimGrid, dimBlock>>>(deviceImageCDF,deviceImageChar, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  //char to float
  dimGrid = dim3((imageWidth+BLOCK_SIZE-1)/BLOCK_SIZE, (imageHeight+BLOCK_SIZE-1)/BLOCK_SIZE, imageChannels);
  dimBlock = dim3(BLOCK_SIZE,BLOCK_SIZE,1);
  charToFloat<<<dimGrid, dimBlock>>>(deviceImageChar, deviceImageFloat, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  
  cudaMemcpy(outputImageData, deviceImageFloat, imageWidth*imageHeight*imageChannels*sizeof(float), cudaMemcpyDeviceToHost);
  
  wbSolution(args, outputImage);
  //@@ insert code here
  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceImageFloat);
  wbTime_stop(GPU, "Freeing GPU Memory");

  free(inputImage);
  free(outputImage);
  return 0;
}
