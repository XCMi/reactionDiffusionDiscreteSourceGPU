/*
  Copyright © Computing Energetic - Calcul Energetique 2020
*/
#define REACTIVE
#pragma once
#include "core.hpp"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <sstream>
#include <deque>
#include <queue>
#include <ctime>
#include <cmath>
#include <typeinfo>
#include <limits>
#include <stdio.h>
#include <stddef.h>
#include <signal.h>
#include "boost/thread.hpp"
#include "Matrix.hpp"
#include "MatrixOperations.hpp"

#define USE_GL

#include "grid.hpp"
#include "particle.hpp"

template<Processor P>
struct Mesh {
  typedef Mesh3D<real, P, NUMBER_VARIABLES> type;
};

StridedCell<real, NUMBER_VARIABLES> typedef Cell;

template<Processor P>
struct Particles {
  typedef Particles3D<real, P, NUMBER_VARIABLES_particle> type;
};

// Gas properties /////////////////////////////////////////////////////

// Thermal diffusivity
const real alpha = 1.0;

//Mass diffusivity
const real D = 0.0;

/////////////////////////////////////////////////////////////////////////

// Reaction model constants /////////////////////////////////////////////

// Activation energy
 const real Ea = 8.0;

// Reaction time
const real tR = 0.5;

// Ignition temperature 
const real Tign = 0.2;

// Pre-exponential factor
const real B = 1.0;

// Dimensionless energy release
const real Q0 = 1.0;

// Initial temperature
const real T0 = 0.0;

// Adiabatic flame temperature
const real Tf = 1.0;

// Wall temperature
const real Twall = 0.0;

// Initial oxidizer concentration
const real C0 = 1.0;

// Final oxidzer concentration
const real Cf = 0.0;

//Initial particle mass
const real MP0 = 1.0;

/////////////////////////////////////////////////////////////////////////
const real X1 = 1.0;
/////////////////////////////////////////////////////////////////////////

const real Tol = 1.0e-8;
const int plot_interval = 1;
int plot_count = plot_interval;

#include "boundaryconditions.hpp"
#include "flux.hpp"
#include "diffusionflux.hpp"
#include "Solver.hpp"
#include "initialconditions.hpp"
#include "initialconditions_particles.hpp"
// #include "render.hpp"
// #include "opengl.hpp"
#include "source.hpp"
#include "particles_source.hpp"
#include "particles_update.hpp"
/*
struct ImageOutputs {
  std::string prefix;
  int plotVariable;
  ColourMode colourMode;
  real min;
  real max;
};
*/
#include "kernels/boundaryconditions.cpp"
// #include "boundaryconditions.cpp"
#include "kernels/flux.cpp"
#include "kernels/diffusionflux.cpp"
#include "kernels/source.cpp"
#include "kernels/particles_source.cpp"
#include "kernels/particles_update.cpp"
// #include "kernels/find_minimum.ipp"
