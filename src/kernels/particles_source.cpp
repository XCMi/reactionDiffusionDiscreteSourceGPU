/*
  Copyright © Computing Energetic - Calcul Energetique 2020
*/
#include "particles_source.hpp"
/*
#if __CUDA_ARCH__ < 600
__device__ double atomicAdd1(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif
*/
//////////////////////////////////////////////////////////////////////////////////////////////

__global__ void sourceParticlesKernel(Mesh<GPU>::type u, Particles<GPU>::type Xp, real dt) {

  const int pi = blockIdx.x * blockDim.x + threadIdx.x;
  const real dx = u.dx();
  const real dy = u.dy();
  const real dz = u.dz();
  const real xpi = Xp(pi, XPOS);
  const real ypi = Xp(pi, YPOS);
  const real zpi = Xp(pi, ZPOS);	
  real ipi, jpi, kpi;
  int Ipi, Jpi, Kpi;
  ipi = ( xpi - dx / 2.0 - u.xMin()) / dx;
  jpi = ( ypi - dy / 2.0 - u.yMin()) / dy;
  kpi = ( zpi - dz / 2.0 - u.zMin()) / dz;
  Ipi = (int) ipi; 
  Jpi = (int) jpi;
  Kpi = (int) kpi;

/*
  if (pi == 0) {
    printf("%f %f\n", xpi, ypi);
    printf("%i %i\n", Ipi, Jpi);
  }
*/

  if (u.active(Ipi, Jpi, Kpi) && Xp.active(pi)) {

    // double dTG000, dTG100, dTG010, dTG001, dTG110, dTG101, dTG011, dTG111;
    double dQ;

    if ( (Xp(pi, TP) > Tign ) && (Xp(pi, MP) > Tol) ) {

      const real dv = dx * dy * dz; 
      const real x0 = xpi - u.x(Ipi);
      const real y0 = ypi - u.y(Jpi);
      const real z0 = zpi - u.z(Kpi);
      const real x1 = dx - x0;
      const real y1 = dy - y0;
      const real z1 = dz - z0;

      const real v000 = x0 * y0 * z0;
      const real v100 = x1 * y0 * z0;
      const real v010 = x0 * y1 * z0;
      const real v001 = x0 * y0 * z1;
      const real v110 = x1 * y1 * z0;
      const real v101 = x1 * y0 * z1;
      const real v011 = x0 * y1 * z1;
      const real v111 = x1 * y1 * z1;

// Ignition-temperature (Boxcar) reaction model ///////////////////////////////////////////////////////     
      if (pi == 0 || tR < Tol) {
        // tau_c = 0
        dQ = MP0 * Q0 / dv;      
        Xp(pi, MP) = 0.0;
      } else {
        // finite tau_c
        dQ = dt * ( MP0 / tR ) * Q0 / dv;      
        Xp(pi, MP) -= dt * ( MP0 / tR );
      }
      
      if (Xp(pi, TAU) > u.time() + Tol) {
        Xp(pi, TAU) = u.time();
      }
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
    real *p11 = u.getAddress(Ipi, Jpi, TEMP);
    real *p21 = u.getAddress(Ipi + 1, Jpi, TEMP);
    real *p12 = u.getAddress(Ipi, Jpi + 1, TEMP);
    real *p22 = u.getAddress(Ipi + 1, Jpi + 1, TEMP);


   atomicAdd1( (double*)u.getAddress(Ipi, Jpi, TEMP), TEMP1 );
   atomicAdd1( (double*)u.getAddress(Ipi + 1, Jpi, TEMP), TEMP2 );
   atomicAdd1( (double*)u.getAddress(Ipi, Jpi + 1, TEMP), TEMP3 );
   atomicAdd1( (double*)u.getAddress(Ipi + 1, Jpi + 1, TEMP), TEMP4);

    __syncthreads();
*/
     u(Ipi, Jpi, Kpi, TG) += dQ * v111 / dv;
     u(Ipi+1, Jpi, Kpi, TG) += dQ * v011 / dv;
     u(Ipi, Jpi+1, Kpi, TG) += dQ * v101 / dv;
     u(Ipi, Jpi, Kpi+1, TG) += dQ * v110 / dv;
     u(Ipi+1, Jpi+1, Kpi, TG) += dQ * v001 / dv;
     u(Ipi+1, Jpi, Kpi+1, TG) += dQ * v010 / dv;
     u(Ipi, Jpi+1, Kpi+1, TG) += dQ * v100 / dv;
     u(Ipi+1, Jpi+1, Kpi+1, TG) += dQ * v000 / dv;
   
    }
 } 

}




