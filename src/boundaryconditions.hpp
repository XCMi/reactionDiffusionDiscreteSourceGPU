#pragma once

template<int BOUNDARY>
__global__ void setBoundaryConditionsKernel(Mesh<GPU>::type u, int BC);

/*
template<bool XDIR>
__global__ void setSpecialBoundaryConditionsKernel(Mesh<GPU>::type u);
*/
