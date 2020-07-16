/*
  Copyright © Computing Energetic - Calcul Energetique 2020
*/
#include "diffusionflux.hpp"

template<int blockDimX, int blockDimY, int blockDimZ>
__global__ void getMixedDiffusionFluxesKernel(Mesh<GPU>::type u, Mesh<GPU>::type flux, const real dt) {

  const int i = blockDimX * blockIdx.x + threadIdx.x - u.ghostCells();
  const int j = blockDimY * blockIdx.y + threadIdx.y - u.ghostCells();
  const int k = blockDimZ * blockIdx.z + threadIdx.z - u.ghostCells();

  const real dx = u.dx();
  const real dy = u.dy();
  const real dz = u.dz();

  // __shared__ real TG_shared[NUMBER_VARIABLES][blockDimZ][blockDimY][blockDimX];

  if (u.within(i, j, k, 0)) {
/*
    // read in the centre point
    for (int k = 0; k < NUMBER_VARIABLES; k++) TG_shared[k][threadIdx.x][threadIdx.y] = u(i, j, k);
    __syncthreads();
*/
    if (u.within(i, j, k, 1)) {
/*
      if ((threadIdx.x > 0) && (threadIdx.x < (blockDimX - 1)) && (threadIdx.y > 0) && (threadIdx.y < (blockDimY - 1))) {

        flux(i, j, TG) = dt * alpha * ( (TG_shared[TG][threadIdx.x + 1][threadIdx.y] - 2.0 * TG_shared[TG][threadIdx.x][threadIdx.y] + TG_shared[TG][threadIdx.x - 1][threadIdx.y]) / (dx * dx)
                           + (TG_shared[TG][threadIdx.x][threadIdx.y + 1] - 2.0 * TG_shared[TG][threadIdx.x][threadIdx.y] + TG_shared[TG][threadIdx.x][threadIdx.y - 1]) / (dy * dy) );   

        flux(i, j, CG) = dt * D * ( (TG_shared[CG][threadIdx.x + 1][threadIdx.y] - 2.0 * TG_shared[CG][threadIdx.x][threadIdx.y] + TG_shared[CG][threadIdx.x - 1][threadIdx.y]) / (dx * dx)
                           + (TG_shared[CG][threadIdx.x][threadIdx.y + 1] - 2.0 * TG_shared[CG][threadIdx.x][threadIdx.y] + TG_shared[CG][threadIdx.x][threadIdx.y - 1]) / (dy * dy) );    

       } else {
*/
        flux(i, j, k, TG) = dt * alpha * (  
                           (u(i + 1, j, k, TG) - 2.0 * u(i, j, k, TG) + u(i - 1, j, k, TG) ) / (dx * dx) +
                           (u(i, j + 1, k, TG) - 2.0 * u(i, j, k, TG) + u(i, j - 1, k, TG) ) / (dy * dy) +
                           (u(i, j, k + 1, TG) - 2.0 * u(i, j, k, TG) + u(i, j, k - 1, TG) ) / (dz * dz)
                            );   

        flux(i, j, k, CG) = dt * D * (  
                           (u(i + 1, j, k, CG) - 2.0 * u(i, j, k, CG) + u(i - 1, j, k, CG) ) / (dx * dx) +
                           (u(i, j + 1, k, CG) - 2.0 * u(i, j, k, CG) + u(i, j - 1, k, CG) ) / (dy * dy) +
                           (u(i, j, k + 1, CG) - 2.0 * u(i, j, k, CG) + u(i, j, k - 1, CG) ) / (dz * dz)
                            );   
/*
       }
*/
       __syncthreads();

    }
  }
}


