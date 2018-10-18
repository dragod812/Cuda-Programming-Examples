#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <helper_cuda.h>
#include <list> 
#include <sstream>
#include <fstream>
#include <string> 
#include <stdio.h>
#include <iostream> 

#define FULL_MASK 0xffffffff

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

class Points
{
  float *X;
  float *Y;

public:
  __host__ __device__ Points() : X(NULL), Y(NULL) {}

  __host__ __device__ Points( float *x, float *y ) : X(x), Y(y) {}

  __host__ __device__ __forceinline__ float2 getPoint( int idx ) const
  {
    return make_float2( X[idx], Y[idx] );
  }

  __host__ __device__ __forceinline__ void setPoint( int idx, const float2 &p ) 
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
		xMin = -700;
		yMin = -700;
		xMax = 700;
		yMax = 700;
	}
	__host__ __device__ void computeCenter( float2 &center ){
		center.x = 0.5f * ( m_p_min.x + m_p_max.x );
		center.y = 0.5f * ( m_p_min.y + m_p_max.y );
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
	//startIdx of points in the bb for the global data array
	int startIdx, endIdx;	
	Quadtree_Node *NE, *NW, *SW, *SE; 
public:
	__host__ __device__ Quadtree_Node() : idx(-1), startIdx(-1), endIdx(-1), NE(NULL), NW(NULL), SW(NULL), SE(NULL){

	}
	__host__ __device__ bool isNull(){
		return (idx == -1);
	}
	__host__ __device__ void setIdx(int idx){
		this->idx = idx;		
	}
	__host__ __device__ int getIdx(){
		return idx;
	}
	__host__ __device__ void setBoundingBox(float x,float y,float X,float Y){
		bb.set(x, y, X, Y);	
	}
	__host__ __device__ __forceinline__ Bounding_Box& getBoundingBox(){
		return bb;
	}
	__host__ __device__ void setRange(int s, int e){
		startIdx = s;
		endIdx = e;
	}
	__host__ __device__ __forceinline__ Quadtree_Node* getSW(){
		return SW;
	}
	__host__ __device__ __forceinline__ Quadtree_Node* getSE(){
		return SE;
	}
	__host__ __device__ __forceinline__ Quadtree_Node* getNW(){
		return NW; 
	}
	__host__ __device__ __forceinline__ Quadtree_Node* getNE(){
		return NE; 
	}
	__host__ __device__ __forceinline__ void setSW( Quadtree_Node* ptr){
		SW = ptr;
	}
	__host__ __device__ __forceinline__ void setNW( Quadtree_Node* ptr){
		NW = ptr;
	}
	__host__ __device__ __forceinline__ void setSE( Quadtree_Node* ptr){
		SE = ptr;
	}
	__host__ __device__ __forceinline__ void setNE( Quadtree_Node* ptr){
		NE = ptr;
	}

	__host__ __device__ __forceinline__ int getStartIdx(){
		return startIdx;
	}
	__host__ __device__ __forceinline__ int getEndIdx(){
		return endIdx;
	}
 	__host__ __device__ __forceinline__ int numberOfPoints(){
		return endIdx - startIdx + 1;
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
    unsigned seed = hash( blockidx.x*blockdim.x + threadidx.x );
    thrust::default_random_engine rng(seed);
    thrust::random::uniform_real_distribution<float> distrib;
    return thrust::make_tuple( distrib(rng), distrib(rng) );
  }
};

class Parameters
{
	const int min_points_per_node;
	//Introduced to minimise shifting of points
	//can have values only 0 and 1 based on slot
	//points[points_slot] is input slot
	//points[(points_slot+1)%2] is output slot
	int points_slot;
	__host__ __device__ Parameters( int mppn ) : min_points_per_node(mppn), points_slot(0) {}
	//copy constructor for the evaluation of children of current node
	__host__ __device__ Parameters( Parameters prm ) : min_points_per_node(prm.min_points_per_node), points_slot((prm.points_slot+1)%2) {}

}

template< int NUM_THREADS_PER_BLOCK >
__global__ 
void buildQuadtree( Quadtree_Node *root, Points *points, Parameters prmtrs){
	const int NUM_WARPS_PER_BLOCK = NUM_THREADS_PER_BLOCK / warpSize;

	//shared memory
	extern __shared__ int smem[];
	
	//warp_id and lane_id
	const int warp_id = threadIdx.x / warpSize;
	const int lane_id = threadIdx.x % warpSize;
	
	// Addresses of shared Memory
	volatile int *s_num_pts[4];
	for( int i = 0 ; i < 4 ; ++i )
		s_num_pts[i] = (volatile int *) &smem[i*NUM_WARPS_PER_BLOCK];

	int lane_mask_lt = (1 << lane_id) - 1; 
	
	int NUM_POINTS = root->numberOfPoints();

	//stop recursion if num_points <= minimum number of points required for recursion 
	if( NUM_POINTS <= prmtrs.min_points_per_node){

		//unable to understand the use of point_selector
		return;
	}

	//get Center of the bounding box
	float2 center;
	const Bounding_Box &box = root->getBoundingBox();
	box.computeCenter( center );

	int NUM_POINTS_PER_WARP = max( warpSize, ( NUM_POINTS + NUM_WARPS_PER_BLOCK - 1 ) / NUM_WARPS_PER_BLOCK );
	
	int warp_begin = root->getStartIdx() + warp_id*NUM_POINTS_PER_WARP;
	int warp_end = min(warp_begin + NUM_POINTS_PER_WARP, root->getEndIdx());

	if( lane_id == 0 )
	{
		s_num_pts[0][warp_id] = 0;
		s_num_pts[1][warp_id] = 0;
		s_num_pts[2][warp_id] = 0;
		s_num_pts[3][warp_id] = 0;
	}
	
	//input points
	const Points &input = points[prmtrs.points_slot];
	
	//__any_sync(unsigned mask, predicate):
		//Evaluate predicate for all non-exited threads in mask and return non-zero if and only if predicate evaluates to non-zero for any of them.
	//count points in each warp that belong to which child
	for( int itr = warp_begin + lane_id ; __any_sync(FULL_MASK, itr < warp_end ) ; itr += warpSize){
		bool is_active = itr < warp_end;

		//get the coordinates of the point
		float2 curP;
		if(is_active)
			curP = input.getPoint(itr);
		else
			curP = make_float2(0.0f, 0.0f);

		//consider standard anticlockwise quadrants for numbering 0 to 3

		//__ballot_sync(unsigned mask, predicate):
			//Evaluate predicate for all non-exited threads in mask and return an integer whose Nth bit is set if and only if predicate evaluates to non-zero for the Nth thread of the warp and the Nth thread is active.
		//__popc
			//Count the number of bits that are set to 1 in a 32 bit integer.
		//top-right Quadrant (Quadrant - I)
		int cnt = __popc( __ballot_sync(FULL_MASK, is_active && curP.x >= center.x && curP.y >= center.y));
		if( cnt > 0 && lane_id == 0 )
			s_num_pts[0][warp_id] += cnt;

		//top-left Quadrant (Quadrant - II)
		cnt = __popc( __ballot_sync(FULL_MASK, is_active && curP.x < center.x && curP.y >= center.y));
		if( cnt > 0 && lane_id == 0 )
			s_num_pts[1][warp_id] += cnt;

		//bottom-left Quadrant (Quadrant - III)
		cnt = __popc( __ballot_sync(FULL_MASK, is_active && curP.x < center.x && curP.y < center.y));
		if( cnt > 0 && lane_id == 0 )
			s_num_pts[2][warp_id] += cnt;

		//bottom-right Quadrant (Quadrant - IV)
		cnt = __popc( __ballot_sync(FULL_MASK, is_active && curP.x >= center.x && curP.y < center.y));
		if( cnt > 0 && lane_id == 0 )
			s_num_pts[3][warp_id] += cnt;
	}		

	//sychronize warps
	//__syncthreads() acts as a barrier at which all threads in the block must wait before any is allowed to proceed
	__syncthreads();

	// Scan the warps' results to know the "global" numbers.
	// First 4 warps scan the numbers of points per child (inclusive scan).
	if( warp_id < 4 )
	{
		int num_pts = lane_id < NUM_WARPS_PER_BLOCK ? s_num_pts[warp_id][lane_id] : 0;
		#pragma unroll
		for( int offset = 1 ; offset < NUM_WARPS_PER_BLOCK ; offset *= 2 )
		{
			int n = __shfl_up_sync( num_pts, offset, NUM_WARPS_PER_BLOCK );
			if( lane_id >= offset )
				num_pts += n;
		}
		if( lane_id < NUM_WARPS_PER_BLOCK )
			s_num_pts[warp_id][lane_id] = num_pts;
	}
	__syncthreads();
	// Compute global offsets.
	if( warp_id == 0 )
	{
		int sum = s_num_pts[0][NUM_WARPS_PER_BLOCK-1];
		for( int row = 1 ; row < 4 ; ++row )
		{
			int tmp = s_num_pts[row][NUM_WARPS_PER_BLOCK-1];
			if( lane_id < NUM_WARPS_PER_BLOCK )
				s_num_pts[row][lane_id] += sum;
			sum += tmp;
		}
	}
	__syncthreads();

}
int main()
{
	//parameters
	const int max_depth = 8;
	const int min_points_per_node = 20;
	int num_points = -1;

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
		  std::cout << "Running on GPU: " << i << " (" << properties.name << ")" << std::endl;
		  std::cout << "Warp Size: " << warp_size << std::endl;
		  std::cout << "Threads Per Block: " << properties.maxThreadsPerBlock<< std::endl;
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
	//Read Points from file and put it into x0(X points) and y0(Y Points)
	std::list<float> stlX, stlY;
	std::ifstream source("2.5width_4patels.txt");
	if(source.is_open()){
		int i = 0;
		for(std::string line;std::getline(source, line); i+=1)   //read stream line by line
		{
			std::istringstream in(line);      
			float x, y;
			in >> x >> y;       
			stlX.push_back(x);
			stlY.push_back(y);
		}
	}
	else{
		printf("No");
		exit(1);
	}

	num_points = stlX.size();	
	thrust::device_vector<float> x0( stlX.begin(), stlX.end() ); 
	thrust::device_vector<float> y0( stlY.begin(), stlY.end() );
	thrust::device_vector<float> x1( num_points );
	thrust::device_vector<float> y1( num_points );

	std::cout << num_points << std::endl;	
	
	//copy pointers to the points into the device because kernels don't support device_vector as input they accept raw_pointers
	//Thrust data types are not understood by a CUDA kernel and need to be converted back to its underlying pointer. 
	//host_points
	Points h_points[2];
	h_points[0].set( thrust::raw_pointer_cast( &x0[0] ), thrust::raw_pointer_cast( &y0[0] ) );
	h_points[1].set( thrust::raw_pointer_cast( &x1[0] ), thrust::raw_pointer_cast( &y1[0] ) );

	//device_points
	Points *d_points;
	checkCudaErrors( cudaMalloc( (void**) &d_points, 2*sizeof(Points) ) ); 
	checkCudaErrors( cudaMemcpy( d_points, h_points, 2*sizeof(Points), cudaMemcpyHostToDevice ) );
	//Setting Cuda Heap size for dynamic memory allocation	
	size_t size = 1024*1024*1024;
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, size);
	cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);

	//Copy root node from host to device
	Quadtree_Node h_root;
	h_root.setRange(0, num_points);
	Quadtree_Node* d_root;
	checkCudaErrors( cudaMalloc( (void**) &d_root, sizeof(Quadtree_Node)));
	checkCudaErrors( cudaMemcpy( d_root, &h_root, sizeof(Quadtree_Node), cudaMemcpyHostToDevice));

	//set the recursion limit based on max_depth
	//maximum possible depth is 24 levels
  	cudaDeviceSetLimit( cudaLimitDevRuntimeSyncDepth, max_depth );

	getchar();
    return 0;
}

