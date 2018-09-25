#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <helper_cuda.h>

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

class Points
{
  float *X;
  float *Y;

public:
  __host__ __device__ Points() : X(NULL), Y(NULL) {}

  __host__ __device__ Points( float *x, float *y ) : X(x), Y(y) {}

  __host__ __device__ __forceinline__ float2 get_point( int idx ) const
  {
    return make_float2( X[idx], Y[idx] );
  }

  __host__ __device__ __forceinline__ void set_point( int idx, const float2 &p ) 
  {
    X[idx] = p.x;
    Y[idx] = p.y;
  }

  __host__ __device__ __forceinline__ void set( float *x, float *y )
  {
    X = x;
    Y = y;
  }
};


class Bounding_Box{
    float xMin , xMax, yMin, yMax;
public:
	__host__ __device__ Bounding_Box(){
		xMin = 0;
		yMin = 0;
		xMax = 1920;
		yMax = 1080;
	}
	__host__ __device__ __forceinline__ float getxMax() const {
		return xMax;		
	}
	__host__ __device__ __forceinline__ float getyMax() const {
		return yMax;		
	}
	__host__ __device__ __forceinline__ float getyMin() const {
		return yMin;		
	}
	__host__ __device__ __forceinline__ float getxMin() const {
		return xMin;		
	}
	__host__ __device__ bool contains(const float2 &p) const {
		return (p.x >= xMin && p.y >= yMin && p.x < xMax && p.y < yMax);
	}
	__host__ __device__ void set(float x, float y, float X, float Y){
		xMin = x;
		yMin = y;
		xMax = X;
		yMax = y;
	}
};

class Quadtree_Node{
	//node index;
	int idx;
	Bounding_Box bb;
	//sIdx of points in the bb for the global data array
	int sIdx, eIdx;	
	__host__ __device__ Quadtree_Node() : idx(-1), sIdx(-1), eIdx(-1){}
	__host__ __device__ bool isNull(){
		return (idx == -1);
	}
	__host__ __device__ void setIdx(int idx){
		this->idx = idx;		
	}
	__host__ __device__ int getIdx(){
		return idx;
	}
	__host__ __device__ void setBoundingBox(int x, int y, int X, int Y){
		bb.set(x, y, X, Y);	
	}
	__host__ __device__ __forceinline__ Bounding_Box& getBoundingBox(){
		return bb;
	}
	__host__ __device__ void setRange(int s, int e){
		sIdx = s;
		eIdx = e;
	}
	/*
	__host__ __device__ __forceinline__ int getSWChildIdx(){
		return 2*idx + 1;
	}
	__host__ __device__ __forceinline__ int getSEChildIdx(){
		return 2*idx + 2;
	}
	__host__ __device__ __forceinline__ int getNWChildIdx(){
		return 2*idx + 3;
	}
	__host__ __device__ __forceinline__ int getNEChildIdx(){
		return 2*idx + 4;
	}
	__host__ __device__ __forceinline__ int getStartIdx(){
		return sIdx;
	}
	__host__ __device__ __forceinline__ int getEndIdx(){
		return eIdx;
	}
	*/
	__host__ __device__ __forceinline__ int numberOfPoints(){
		return eIdx - sIdx + 1;
	}
};

struct Random_generator
{
  __host__ __device__ unsigned int hash(unsigned int a)
  {
      a = (a+0x7ed55d16) + (a<<12);
      a = (a^0xc761c23c) ^ (a>>19);
      a = (a+0x165667b1) + (a<<5);
      a = (a+0xd3a2646c) ^ (a<<9);
      a = (a+0xfd7046c5) + (a<<3);
      a = (a^0xb55a4f09) ^ (a>>16);
      return a;
  }

  __host__ __device__ __forceinline__ thrust::tuple<float, float> operator()() 
  {
    unsigned seed = hash( blockIdx.x*blockDim.x + threadIdx.x );
    thrust::default_random_engine rng(seed);
    thrust::random::uniform_real_distribution<float> distrib;
    return thrust::make_tuple( distrib(rng), distrib(rng) );
  }
};


int main()
{

	//Set Cuda Device
  	int device_count = 0, device = -1, warp_size = 0;
  	checkCudaErrors( cudaGetDeviceCount( &device_count ) );
	for( int i = 0 ; i < device_count ; ++i )
	{
		cudaDeviceProp properties;
		checkCudaErrors( cudaGetDeviceProperties( &properties, i ) );
		if( properties.major > 3 || ( properties.major == 3 && properties.minor >= 5 ) )
		{
		  device = i;
		  warp_size = properties.warpSize;
		  std::cout << "Running on GPU " << i << " (" << properties.name << ")" << std::endl;
		  break;
		}
		std::cout << "GPU " << i << " (" << properties.name << ") does not support CUDA Dynamic Parallelism" << std::endl;
	}
	if( device == -1 )
	{
		//cdpQuadTree requires SM 3.5 or higher to use CUDA Dynamic Parallelism.  Exiting...
		exit(EXIT_SUCCESS);
	}
	cudaSetDevice(device);
	getchar();

    return 0;
}


