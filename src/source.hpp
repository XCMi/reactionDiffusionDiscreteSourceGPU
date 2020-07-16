#pragma once

template<int blockDimX, int blockDimY, int blockDimZ>
__global__ void sources(Mesh<GPU>::type u, const real dt);

