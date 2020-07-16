/*
  Copyright © Computing Energetic - Calcul Energetique 2020
*/
#pragma once

__global__ void setInitialConditions(Mesh<GPU>::type u) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x,
            j = blockIdx.y * blockDim.y + threadIdx.y,
            k = blockIdx.z * blockDim.z + threadIdx.z;

  if (u.active(i, j, k)) {
	  
    const real x = u.x(i), y = u.y(j), z = u.z(k);
    real T = 0.0, C = 0.0;
    
    if (  x <= X1 ) {
      T += T0;
      C += C0;
    } else {
      T += T0;
      C += C0;
    }
     
    u(i, j, k, TG) = T;
    u(i, j, k, CG) = C;
  
  }
}

