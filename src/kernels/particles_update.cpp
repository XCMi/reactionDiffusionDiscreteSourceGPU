#include "particles_update.hpp"

__device__ __host__ real Bilinear_Interpolation(const real dx, const real dy, const real x, const real y, const real Q1, const real Q2, const real Q3, const real Q4) {
const real l1 = sqrt ( x * x + y * y );
const real l2 = sqrt ( (dx - x) * (dx - x) + y * y );
const real l3 = sqrt ( (dy - y) * (dy - y) + x * x );
const real l4 = sqrt ( (dx - x) * (dx - x) + (dy - y) * (dy - y) );
return ( (Q1 / l1) + (Q2 / l2) + (Q3 / l3) + (Q4 / l4) ) / ( 1.0 / l1 + 1.0 / l2 + 1.0 / l3 + 1.0 / l4 );
}

__device__ __host__ real Trilinear_Interpolation(const real dx, const real dy, const real dz, 
const real x0, const real y0, const real z0,
const real q000, const real q100, const real q010, const real q001,
const real q110, const real q101, const real q011, const real q111) {

const real dv = dx * dy * dz;
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

const real result = ( q000*v111 + q100*v011 + q010*v101 + q001*v110 + q110*v001 + q101*v010 + q011*v100 + q111*v000 ) / dv;

printf("x0 = %f, y0 = %f, z0 = %f, x1 = %f, y1 = %f, z1 = %f\n", x0, y0, z0, x1, y1, z1);
printf("result = %f, q000 = %f, q100 = %f, q010 = %f, q001 = %f, q110 = %f, q101 = %f, q011 = %f, q111 = %f\n", result, q000, q100, q010, q001, q110, q101, q011, q111);
printf("dv = %f, v000 = %f, v100 = %f, v010 = %f, v001 = %f, v110 = %f, v101 = %f, v011 = %f, v111 = %f\n", dv, v000, v100, v010, v001, v110, v101, v011, v111);

return result;
}

//****************************************************************************

__global__ void updateParticlesKernel(Mesh<GPU>::type u, Particles<GPU>::type Xp) {

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
    printf("Tp = %f\n", Xp(pi, TP));
  }
*/
  if (u.active(Ipi, Jpi, Kpi) && Xp.active(pi)) {
   
    const real x = xpi - u.x(Ipi);
    const real y = ypi - u.y(Jpi);
    const real z = zpi - u.z(Kpi);
    
    real p000_local[NUMBER_VARIABLES], 
         p100_local[NUMBER_VARIABLES], 
         p010_local[NUMBER_VARIABLES], 
         p001_local[NUMBER_VARIABLES],
         p110_local[NUMBER_VARIABLES], 
         p101_local[NUMBER_VARIABLES], 
         p011_local[NUMBER_VARIABLES], 
         p111_local[NUMBER_VARIABLES];
    Cell temp000(&p000_local[0], 1), 
         temp100(&p100_local[0], 1), 
         temp010(&p010_local[0], 1), 
         temp001(&p001_local[0], 1),
         temp110(&p110_local[0], 1), 
         temp101(&p101_local[0], 1), 
         temp011(&p011_local[0], 1), 
         temp111(&p111_local[0], 1);
    Cell & p000 = temp000, 
         & p100 = temp100, 
         & p010 = temp010, 
         & p001 = temp001,
         & p110 = temp110, 
         & p101 = temp101, 
         & p011 = temp011, 
         & p111 = temp111;

    for (int var = 0; var < CONSERVATIVE_VARIABLES; var++) {

      p000[var] = u(Ipi, Jpi, Kpi, var);
      p100[var] = u(Ipi+1, Jpi, Kpi, var);
      p010[var] = u(Ipi, Jpi+1, Kpi, var);
      p001[var] = u(Ipi, Jpi, Kpi+1, var);
      p110[var] = u(Ipi+1, Jpi+1, Kpi, var);
      p101[var] = u(Ipi+1, Jpi, Kpi+1, var);
      p011[var] = u(Ipi, Jpi+1, Kpi+1, var);
      p111[var] = u(Ipi+1, Jpi+1, Kpi+1, var);

      // Xp( pi, var+2 ) = Bilinear_Interpolation(dx, dy, x, y, p000[var], p100[var], p010[var], p110[var]);
      Xp( pi, var+3 ) = Trilinear_Interpolation(dx, dy, dz, x, y, z, p000[var], p100[var], p010[var], p001[var], p110[var], p101[var], p011[var], p111[var]);
   
    }
/*    
    if (pi == 0) {
      printf("Tp = %f, T000 = %f, T100 = %f, T010 = %f, T001 = %f, T100 = %f, T110 = %f, T101 = %f, T011 = %f, T111 = %f\n", Xp(pi, TP),p000[TG], p100[TG], p010[TG], p001[TG], p110[TG], p101[TG], p011[TG], p111[TG]);
    }
*/    
  } 

}

/*
__global__ void moveParticlesKernel(Particles<GPU>::type Xp, const real dt, const real dtp, Mesh<GPU>::type u) {
  const int pi = blockDim.x * blockIdx.x + threadIdx.x;
  const real ymax = u.yMax();

  Xp(pi, XPOS) += 0.5 * ( 2.0 * dtp + dt ) * Xp( pi, VELXP ) - 0.5 * dt * Xp( pi, VELXPP );
  Xp(pi, YPOS) += 0.5 * ( 2.0 * dtp + dt ) * Xp( pi, VELYP ) - 0.5 * dt * Xp( pi, VELYPP );

  if (Xp(pi, YPOS) > ymax){
    Xp(pi, YPOS) -= ymax;
  } else if (Xp(pi, YPOS) < 0.0){ 
    Xp(pi, YPOS) += ymax;
  }
}
*/
