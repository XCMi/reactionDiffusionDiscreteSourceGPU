/*
  Copyright © Computing Energetic - Calcul Energetique 2020
*/
#pragma once

__global__ void setInitialConditions_particles(Mesh<GPU>::type u, Particles<GPU>::type Xp, const real xMax, const real yMax, const real zMax) {
  const int pi = blockIdx.x * blockDim.x + threadIdx.x;  // Particle index
  const real dx = u.dx();
  const real dy = u.dy();
  const real dz = u.dz();

  real xpi = 0.0, ypi = 0.0, zpi = 0.0, Tpi = 0.0, Cpi = 0.0, Mpi = 0.0, Yfi = 0.0;

  xpi += pi * 1.0 + 0.5;

  if (pi == 0) {
    Tpi += Tign + Tol;
  } else {
    Tpi += T0;
  }

  Cpi += C0;
  Mpi += MP0;

  real ipi, jpi, kpi;
  int Ipi, Jpi, Kpi;
  ipi = ( xpi - dx / 2.0 ) / dx;
  jpi = ( ypi - dy / 2.0 ) / dy;
  kpi = ( zpi - dz / 2.0 ) / dz;
  Ipi = (int) ipi; 
  Jpi = (int) jpi;
  Kpi = (int) kpi;

  if (u.active(Ipi, Jpi, Kpi) && Xp.active(pi)) {

    Xp( pi, XPOS ) = xpi;
    Xp( pi, YPOS ) = ypi;
    Xp( pi, ZPOS ) = zpi;
    Xp( pi, TP ) = Tpi;
    Xp( pi, CP ) = Cpi;
    Xp( pi, MP ) = Mpi;
    Xp( pi, YF ) = Yfi;
    Xp( pi, TAU ) = 1.0e12;
  
  }

}

