#pragma once

__device__ __host__ real Bilinear_Interpolation(const real dx, const real dy, const real x, const real y, const real Q1, const real Q2, const real Q3, const real Q4);

__device__ __host__ real Trilinear_Interpolation(const real dx, const real dy, const real dz, const real x0, const real y0, const real z0, const real q000, const real q100, const real q010, const real q001, const real q110, const real q101, const real q011, const real q111);

__global__ void updateParticlesKernel(Mesh<GPU>::type u, Particles<GPU>::type Xp);

/*
__global__ void moveParticlesKernel(Particles<GPU>::type Xp, Particles<GPU>::type Up, Particles<GPU>::type Upp, const real dt, const real dtp, Mesh<GPU>::type u);
*/

