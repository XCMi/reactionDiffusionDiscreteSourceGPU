/*
  Copyright © Computing Energetic - Calcul Energetique 2020
*/
#include "flux.hpp"

__global__ void addSemiFluxesKernel(Mesh<GPU>::type u, Mesh<GPU>::type flux) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  const int j = blockDim.y * blockIdx.y + threadIdx.y;
  const int k = blockDim.z * blockIdx.z + threadIdx.z;

  if (u.active(i, j, k)) {
    {
      #pragma unroll
      for (int var = 0; var < CONSERVATIVE_VARIABLES + NONCONSERVATIVE_VARIABLES; var++) {
        u(i, j, k, var) += flux(i, j, k, var);
      }
    }
  }
}

