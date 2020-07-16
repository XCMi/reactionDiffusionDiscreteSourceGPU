/*
  Copyright © Computing Energetic - Calcul Energetique 2020
*/
#include "source.hpp"

template<int blockDimX, int blockDimY, int blockDimZ>
__global__ void sources(Mesh<GPU>::type u, const real dt) {
	const int dimX = blockDimX,
	          dimY = blockDimY,
	          dimZ = blockDimZ;
	const int i = dimX * blockIdx.x + threadIdx.x;
	const int j = dimY * blockIdx.y + threadIdx.y;
	const int k = dimZ * blockIdx.z + threadIdx.z;

	if (u.active(i, j, k)) {
		real c[NUMBER_VARIABLES];
		for (int var = 0; var < NUMBER_VARIABLES; var++) {
			c[k] = u(i, j, k, var);
		}
		real W = 0.0;
		if (c[CG] > Tol){
			W = - B * c[CG] * exp( - Ea / c[TG] );
		}  
        c[TG] -= Q0 * W * (1.0 - T0) * dt;
	    c[CG] += W * dt;
    
		for (int var = 0; var < NUMBER_VARIABLES; var++) {
			u(i, j, k, var) = c[var];
		}
    }

}

