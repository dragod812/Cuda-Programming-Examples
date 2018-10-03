//if the number of points not in the form of 2^k
int divup(int a, int b)
{
    if (a % b)  /* does a divide b leaving a remainder? */
        return a / b + 1; /* add in additional block */
    else
        return a / b; /* divides cleanly */
}
int x = 4096, y = 4096;
dim3 blockdim(64, 4), griddim(divup(x, bs.x), divup(y, bs.y));
kernerFunction<<<griddim,blockdim>>>(args)
//If the parent kernel needs results computed by the child kernel to do its own work,
//  it must ensure that the child grid has finished execution before continuing by explicitly 
//  synchronizing using cudaDeviceSynchronize(void)


// __syncthreads() acts as a barrier at which all threads in the block must wait before any is 
// allowed to proceed