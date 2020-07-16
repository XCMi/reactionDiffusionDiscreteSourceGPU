/*
  Copyright © Computing Energetic - Calcul Energetique 2020
*/
#include "boundaryconditions.hpp"

template<int BOUNDARY>
__global__ void setBoundaryConditionsKernel(Mesh<GPU>::type u, int BC) {

  const int i = blockIdx.x * blockDim.x + threadIdx.x - u.ghostCells();
  const int j = blockIdx.y * blockDim.y + threadIdx.y - u.ghostCells();
  
  if (BOUNDARY == LEFT && u.exists(0, i, j)) {
    if (BC == ADIABATIC) {
      u(-1, i, j, TG) = u(0, i, j, TG);
      u(-1, i, j, CG) = u(0, i, j, CG);
    } else if (BC == ISOTHERMAL) {
      u(-1, i, j, TG) = 2.0 * Twall - u(0, i, j, TG);
      u(-1, i, j, CG) = u(0, i, j, CG);
    } else if (BC == PERIODIC) {
      u(-1, i, j, TG) = u(u.activeNx()-1, i, j, TG);
      u(-1, i, j, CG) = u(u.activeNx()-1, i, j, CG);
    }
  }

  if (BOUNDARY == RIGHT && u.exists(0, i, j)) {
    if (BC == ADIABATIC) {
      u(u.activeNx(), i, j, TG) = u(u.activeNx()-1, i, j, TG);
      u(u.activeNx(), i, j, CG) = u(u.activeNx()-1, i, j, CG);
    } else if (BC == ISOTHERMAL) {
      u(u.activeNx(), i, j, TG) = 2.0 * Twall - u(u.activeNx()-1, i, j, TG);
      u(u.activeNx(), i, j, CG) = u(u.activeNx()-1, i, j, CG);
    } else if (BC == PERIODIC) {
      u(u.activeNx(), i, j, TG) = u(0, i, j, TG);
      u(u.activeNx(), i, j, CG) = u(0, i, j, CG);
    }
  }
  
  if (BOUNDARY == BOTTOM && u.exists(i, 0, j)) {
    if (BC == ADIABATIC) {
      u(i, -1, j, TG) = u(i, 0, j, TG);
      u(i, -1, j, CG) = u(i, 0, j, CG);
    } else if (BC == ISOTHERMAL) {
      u(i, -1, j, TG) = 2.0 * Twall - u(i, 0, j, TG);
      u(i, -1, j, CG) = u(i, 0, j, CG);
    } else if (BC == PERIODIC) {
      u(i, -1, j, TG) = u(i, u.activeNy()-1, j, TG);
      u(i, -1, j, CG) = u(i, u.activeNy()-1, j, CG);
    }
  }
  
  if (BOUNDARY == TOP && u.exists(i, 0, j)) {
    if (BC == ADIABATIC) {
      u(i, u.activeNy(), j, TG) = u(i, u.activeNy()-1, j, TG);
      u(i, u.activeNy(), j, CG) = u(i, u.activeNy()-1, j, CG);
    } else if (BC == ISOTHERMAL) {
      u(i, u.activeNy(), j, TG) = 2.0 * Twall - u(i, u.activeNy()-1, j, TG);
      u(i, u.activeNy(), j, CG) = u(i, u.activeNy()-1, j, CG);
    } else if (BC == PERIODIC) {
      u(i, u.activeNy(), j, TG) = u(i, 0, j, TG);
      u(i, u.activeNy(), j, CG) = u(i, 0, j, CG);
    }
  }
  
  if (BOUNDARY == BACK && u.exists(i, j, 0)) {
    if (BC == ADIABATIC) {
      u(i, j, -1, TG) = u(i, j, 0, TG);
      u(i, j, -1, CG) = u(i, j, 0, CG);
    } else if (BC == ISOTHERMAL) {
      u(i, j, -1, TG) = 2.0 * Twall - u(i, j, 0, TG);
      u(i, j, -1, CG) = u(i, j, 0, CG);
    } else if (BC == PERIODIC) {
      u(i, j, -1, TG) = u(i, j, u.activeNz()-1, TG);
      u(i, j, -1, CG) = u(i, j, u.activeNz()-1, CG);
    }
  }
  
  if (BOUNDARY == FRONT && u.exists(i, j, 0)) {
    if (BC == ADIABATIC) {
      u(i, j, u.activeNz(), TG) = u(i, j, u.activeNz()-1, TG);
      u(i, j, u.activeNz(), CG) = u(i, j, u.activeNz()-1, CG);
    } else if (BC == ISOTHERMAL) {
      u(i, j, u.activeNz(), TG) = 2.0 * Twall - u(i, j, u.activeNz()-1, TG);
      u(i, j, u.activeNz(), CG) = u(i, j, u.activeNz()-1, CG);
    } else if (BC == PERIODIC) {
      u(i, j, u.activeNz(), TG) = u(i, j, 0, TG);
      u(i, j, u.activeNz(), CG) = u(i, j, 0, CG);
    }
  }

}

/*
template<bool XDIR>
__global__ void setSpecialBoundaryConditionsKernel(Mesh<GPU>::type u) {
  const bool YDIR = !XDIR;

  const int k = blockIdx.x * blockDim.x + threadIdx.x - u.ghostCells();

  if (XDIR && u.exists(k, 0) && k < u.i(XCORNER)) {
    const int j = u.j(YCORNER);
    for (int n = 0; n < 2; n++) {
      u(k, j + n + 1) = u(k, j - n);
      u(k, j + n + 1, YMOMENTUM) = -u(k, j + n + 1, YMOMENTUM);

    }
  }
  else if (!XDIR && u.exists(0, k) && k > u.j(YCORNER)) {
    const int i = u.i(XCORNER);
    for (int n = 0; n < 2; n++) {
      u(i - n - 1, k) = u(i + n, k);
      u(i - n - 1, k, YMOMENTUM) = -u(i - n - 1, k, YMOMENTUM);
    }
  }
}
*/
