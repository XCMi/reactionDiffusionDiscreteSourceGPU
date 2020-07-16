/*
  Copyright © Computing Energetic - Calcul Energetique 2020
*/
#include <iostream>
#include <iomanip>
#include <stack>
#include <utility>
#include <libconfig.h++>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include "source.hpp"
#include "particles_source.hpp"
#include "particles_update.hpp"
#include "initialconditions.hpp"
#include "initialconditions_particles.hpp"

//////////////////////////////////////////////////////////////////////////////////////
class Solver {
public:
  double timeFluxes, timeDiffusionFluxes, timeSourcing, timeParticles, timeReducing, timeAdding;
  int stepNumber;
  Mesh<GPU>::type* u;
  Mesh<GPU>::type* fluxes;
  Particles<GPU>::type* Xp;
  std::vector<double> outputTimes;
  std::string outputDirectory;
  real targetCFL;
  int outputNumber;
  double lastDt;
  double time_step;
  std::string boundary_conditions[6];
  std::vector<int> BCs;

public:
  enum Status {OK, OUTPUT, FINISHED};

private:
  Solver::Status status;

public:
  Solver(std::string filename) :
    lastDt(0.0),
    time_step(0.0),
    timeFluxes(0.0),
    timeDiffusionFluxes(0.0),
    timeSourcing(0.0),
    timeParticles(0.0),
    timeReducing(0.0),
    timeAdding(0.0),
    stepNumber(0),
    outputNumber(0)
  {
    using namespace libconfig;

    Config config;
    // automatically cast float <-> int
    config.setAutoConvert(true);
    config.readFile(filename.c_str());

    Setting& root = config.getRoot();

    // Parse simulation parameters
    const Setting& simulation = root["simulation"];

    if (simulation.exists("outputDirectory")) {
      outputDirectory = simulation["outputDirectory"].c_str();
    } else {
      outputDirectory = "";
    }

    // Gas domain config
    int Nx, Ny, Nz;
    real xMin, xMax, yMin, yMax, zMin, zMax, dx, dy, dz;
    Nx = simulation["grid"]["cells"]["x"];
    Ny = simulation["grid"]["cells"]["y"];
    Nz = simulation["grid"]["cells"]["z"];   
    xMin = simulation["grid"]["size"]["x"][0];
    xMax = simulation["grid"]["size"]["x"][1];
    yMin = simulation["grid"]["size"]["y"][0];
    yMax = simulation["grid"]["size"]["y"][1];
    zMin = simulation["grid"]["size"]["z"][0];
    zMax = simulation["grid"]["size"]["z"][1];
    targetCFL = simulation["targetCFL"];
    dx = (xMax - xMin) / Nx;
    dy = (yMax - yMin) / Ny;
    dz = (zMax - zMin) / Nz;
    time_step = targetCFL * min (min(dx * dx, dy * dy), dz * dz) / max( alpha, D );
    u = new Mesh<GPU>::type(Nx, Ny, Nz, 1, xMin, xMax, yMin, yMax, zMin, zMax);
    fluxes = new Mesh<GPU>::type(*u, Mesh<GPU>::type::Allocate);

    // Particles config
    int NP;
    NP = simulation["number_of_particles"];
    Xp = new Particles<GPU>::type(NP);
    
    // Read BCs 
    boundary_conditions[LEFT] = simulation["boundary_conditions"]["left"].c_str();
    boundary_conditions[RIGHT] = simulation["boundary_conditions"]["right"].c_str();
    boundary_conditions[BOTTOM] = simulation["boundary_conditions"]["bottom"].c_str();
    boundary_conditions[TOP] = simulation["boundary_conditions"]["top"].c_str();
    boundary_conditions[BACK] = simulation["boundary_conditions"]["back"].c_str();
    boundary_conditions[FRONT] = simulation["boundary_conditions"]["front"].c_str();    
    for (int i = 0; i < 6; i++) {
      if (boundary_conditions[i] == "adiabatic") {
        BCs.push_back(ADIABATIC);
      } else if (boundary_conditions[i] == "isothermal") {
        BCs.push_back(ISOTHERMAL);
      } else if (boundary_conditions[i] == "periodic") {
        BCs.push_back(PERIODIC);
      }
    }

    // Read time squence for output
    if (simulation.exists("start")) {
      real start = simulation["start"];
      real end = simulation["end"];
      real interval = simulation["interval"];
      for (int i = 0; start + i * interval < end + Tol; i++) {
        outputTimes.push_back(start + i * interval);
      }
    } else {
      outputTimes.push_back(1e30);
    }
 
    // Set initial conditions of the gas
    dim3 blockDim(64);
    dim3 gridDim = u->totalCover(blockDim);
    setInitialConditions<<<gridDim, blockDim>>>(*u);
    cudaThreadSynchronize();
    checkForError();

    // Set initial positions of the particles
    dim3 blockDim_p(64);
    dim3 gridDim_p = Xp->totalCover(blockDim_p);
    setInitialConditions_particles<<<gridDim_p, blockDim_p>>>(*u, *Xp, xMax, yMax, zMax);
    cudaThreadSynchronize();
    checkForError();

  }

  Solver::Status step();

  double setBoundaryConditions(std::vector<int> BCs);
  double getDt(real&);
  double getDiffusionFluxes(real dt);
  double addFluxes();
  double updateParticles();
  double checkValidity();
  double source(real dt);
  double sourceParticles(real dt);

  template<bool T>
  double addFluxes();
  double getTimeFluxes() { return timeFluxes; }
  double getTimeParticles() { return timeParticles; }
  double getTimeSourcing() { return timeSourcing; }
  double getTimeReducing() { return timeReducing; }
  double getTimeAdding() { return timeAdding; }
  int getStepNumber() { return stepNumber; }
  int getOutputNumber() { return outputNumber; }
};

Solver::Status Solver::step() {
  float timeFlux = 0.0, timeDiffusionFlux = 0.0, timeParticle = 0.0, timeReduce = 0.0, timeSource = 0.0, timeBCs = 0.0;

  if (u->time() >= outputTimes[outputTimes.size() - 1]) {
    return FINISHED;
  }

  real dt = time_step;
  timeReduce = getDt(dt);
  checkValidity();
  
  timeSource += sourceParticles(dt/2.0);
  timeParticle += updateParticles();
  
  timeBCs = setBoundaryConditions(BCs);
  timeFlux += getDiffusionFluxes(dt);
  timeFlux += addFluxes<true>();
  timeParticle += updateParticles();
  
  timeSource += sourceParticles(dt/2.0);
  timeParticle += updateParticles();

  checkValidity();

  u->time(u->time() + dt);

  timeFluxes += timeFlux;
  timeAdding += timeBCs;
  timeReducing += timeReduce;
  timeSourcing += timeSource;
  timeParticles += timeParticle;

  size_t freeMemory, totalMemory;
  cudaMemGetInfo(&freeMemory, &totalMemory);

  stepNumber++;

  std::cout << "# Step " << stepNumber << "[" << outputNumber << "]: " << std::fixed << std::setprecision(11) << u->time() << " (dt=" << std::setprecision(3) << std::scientific << dt << "). Time {diffusion fluxes=" << std::fixed << std::setprecision(3) << timeFlux << "ms, reduction=" << timeReduce << "ms, BCs=" << timeBCs << "ms, particles=" << timeParticle << "ms}. Memory " << std::fixed << std::setprecision(0) << (float) ((totalMemory - freeMemory) >> 20) << " MiB / " << (float) (totalMemory >> 20) << " MiB" << std::endl;

  return this->status;
}
//////////////////////////////////////////////////////////////////////////////////
double Solver::setBoundaryConditions(std::vector<int> BCs) {
  cudaEvent_t start, stop;
  float time;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  dim3 blockDim(16, 16);
  dim3 gridDim;

  cudaEventRecord(start, 0);
  gridDim.x = 1 + ( u->Ny() - 1) / blockDim.x;
  gridDim.y = 1 + ( u->Nz() - 1) / blockDim.y;
  setBoundaryConditionsKernel<LEFT><<<gridDim, blockDim>>>(*u, BCs[LEFT]);
  cudaThreadSynchronize();
  setBoundaryConditionsKernel<RIGHT><<<gridDim, blockDim>>>(*u, BCs[RIGHT]);
  cudaThreadSynchronize();
  
  gridDim.x = 1 + ( u->Nx() - 1) / blockDim.x;
  gridDim.y = 1 + ( u->Nz() - 1) / blockDim.y;
  setBoundaryConditionsKernel<BOTTOM><<<gridDim, blockDim>>>(*u, BCs[BOTTOM]);
  cudaThreadSynchronize();
  setBoundaryConditionsKernel<TOP><<<gridDim, blockDim>>>(*u, BCs[TOP]);
  cudaThreadSynchronize();
  
  gridDim.x = 1 + ( u->Nx() - 1) / blockDim.x;
  gridDim.y = 1 + ( u->Ny() - 1) / blockDim.y;
  setBoundaryConditionsKernel<BACK><<<gridDim, blockDim>>>(*u, BCs[BACK]);
  cudaThreadSynchronize();
  setBoundaryConditionsKernel<FRONT><<<gridDim, blockDim>>>(*u, BCs[FRONT]);
  cudaThreadSynchronize();
  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  checkForError();
  cudaEventElapsedTime(&time, start, stop);

  return time;
}
//////////////////////////////////////////////////////////////////////////////////
double Solver::getDt(real& dt) {
  cudaEvent_t start, stop;
  float time;

  if (u->time() + dt >= outputTimes[outputNumber]) {
    dt = outputTimes[outputNumber] - u->time();
    outputNumber++;
    status = OUTPUT;
  } else {
    status = OK;
  }

  return time;
}

// Compute diffusive fluxes 
////////////////////////////////////////////////////////////////////////////////////
double Solver::getDiffusionFluxes(real dt) {
  cudaEvent_t start, stop;
  float time;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  dim3 blockDim(8, 8, 8);
  dim3 gridDim = u->totalCover(blockDim);

  getMixedDiffusionFluxesKernel<8, 8, 8><<<gridDim, blockDim>>>(*u, *fluxes, dt);
  cudaThreadSynchronize();

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  checkForError();
  cudaEventElapsedTime(&time, start, stop);

  return time;
}
////////////////////////////////////////////////////////////////////////////////////
template<bool X>
double Solver::addFluxes(void) {
  cudaEvent_t start, stop;
  float time;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  dim3 blockDim(8, 8, 8);
  dim3 gridDim = u->totalCover(blockDim);
  addSemiFluxesKernel<<<gridDim, blockDim>>>(*u, *fluxes);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  checkForError();
  cudaEventElapsedTime(&time, start, stop);

  return time;
}
//////////////////////////////////////////////////////////////////
double Solver::checkValidity(void) {
  cudaEvent_t start, stop;
  float time;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
	  dim3 blockDim(8, 8, 8);
  dim3 gridDim = u->totalCover(blockDim);
  //checkValidityKernel<<<gridDim, blockDim>>>(*u);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  checkForError();
  cudaEventElapsedTime(&time, start, stop);

  return time;
}
//////////////////////////////////////////////////////////////////
double Solver::source(real dt) {
  cudaEvent_t start, stop;
  float time;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  dim3 blockDim(8, 8, 8);
  dim3 gridDim = u->totalCover(blockDim);

  sources<8, 8, 8><<<gridDim, blockDim>>>(*u, dt);
  cudaThreadSynchronize();

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  checkForError();
  cudaEventElapsedTime(&time, start, stop);

  return time;
}
//////////////////////////////////////////////////////////////////
double Solver::sourceParticles(real dt) {
  cudaEvent_t start, stop;
  float time;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  dim3 blockDim(32);
  dim3 gridDim = Xp->totalCover(blockDim);

  sourceParticlesKernel<<<gridDim, blockDim>>>(*u, *Xp, dt);
  cudaThreadSynchronize();

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  checkForError();
  cudaEventElapsedTime(&time, start, stop);

  return time;
}
//////////////////////////////////////////////////////////////////
double Solver::updateParticles() {
  cudaEvent_t start, stop;
  float time;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  dim3 blockDim(32);
  dim3 gridDim = Xp->totalCover(blockDim);

  updateParticlesKernel<<<gridDim, blockDim>>>(*u, *Xp);
  cudaThreadSynchronize();

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  checkForError();
  cudaEventElapsedTime(&time, start, stop);

  return time;
}
