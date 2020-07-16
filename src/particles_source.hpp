#pragma once

/*#if __CUDA_ARCH__ < 600
__device__ real atomicAdd(real* address, real val);
#endif
*/
__global__ void sourceParticlesKernel(Mesh<GPU>::type u, Particles<GPU>::type Xp, real dt); 
