
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
	__host__ __device__ Bouding_Box(){
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
}

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
	__host__ __device__ void setBoundingBox(int x, int y, int X, int Y){
		bb.set(x, y, X, Y);	
	}
	__host__ __device__ void setRange(int s, int e){
		sIdx = s;
		eIdx = e;
	}
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
	__host__ __device__ __forceinline__ int numberOfPoints(){
		return eIdx - sIdx + 1;
	}

}

int main()
{
    
    return 0;
}


