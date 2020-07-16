/*
  Copyright © Computing Energetic - Calcul Energetique 2020
*/
#include "ghost.hpp"
#include <sys/times.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


bool halt = false;

void signalHandler(int signal = 0) {
  halt = true;
}


// Main function on CPU /////////////////////////////////////////////////////////
int main(int argc, char** argv) {

  // capture SIGINT
  signal(SIGINT, signalHandler);

  if (argc < 2) {
    std::cerr << "Invoke with " << argv[0] << " <configuration file>" << std::endl;
    exit(1);
  }

  // select a device
  int num_devices;
  cudaGetDeviceCount(&num_devices);
  std::cout << "#Found " << num_devices << " GPGPUs" << std::endl;
  cudaDeviceProp properties;
  int best_device = 0;
  if (num_devices > 1) {
    // if there's more than one, pick the one with the highest compute capability
    int best_computemode = 0, computemode;
    for (int device = 0; device < num_devices; device++) {
      cudaGetDeviceProperties(&properties, device);
      std::cout << "  #" << device << " " << properties.name << ": " << properties.multiProcessorCount << " processors, compute capability " << properties.major << "." << properties.minor << std::endl;
      computemode = properties.major << 4 + properties.minor;
      if (best_computemode < computemode) {
        best_computemode = computemode;
        best_device = device;
      }
    }
  }
        best_device = atoi(argv[2]);
  cudaGetDeviceProperties(&properties, best_device);
  std::cout << "#  using #" << best_device << " (" << properties.name << ")" << std::endl;
  cudaSetDevice(best_device);

  // start a timer to get the total wall time at the end of the run
  struct tms startTimes, endTimes;
  timespec startClock, endClock;
  times(&startTimes);
  clock_gettime(CLOCK_REALTIME, &startClock);

  Solver solver(argv[1]);

  Solver::Status status = Solver::OUTPUT;

  Mesh<CPU>::type uCPU(*solver.u, Mesh<CPU>::type::Allocate);
  Particles<CPU>::type uCPU_p(*solver.Xp, Particles<CPU>::type::Allocate);

  // Randomize particle positions /////////////////////////////////////////// 
/*
  time_t t_rand;
  srand((unsigned) time(&t_rand));

  for (int i = 0; i < uCPU_p.NP(); ++i)  { 

    int rand_i_int = rand() % 10000;                         
    float rand_i_real = rand_i_int/10000.0;
    uCPU_p(i, XPOS) = rand_i_real * (uCPU.xMax() - X1) + X1;

    int rand_j_int = rand() % 10000;                         
    float rand_j_real = rand_j_int/10000.0;
    uCPU_p(i, YPOS) = rand_j_real * uCPU.yMax();
    
    int rand_k_int = rand() % 10000;                         
    float rand_k_real = rand_k_int/10000.0;
    uCPU_p(i, ZPOS) = rand_k_real * uCPU.zMax();

    uCPU_p(i, TP) = T0; 
    uCPU_p(i, CP) = C0; 
    uCPU_p(i, MP) = MP0; 

  }

  *solver.Xp = uCPU_p;
*/
///////////////////////////////////////////////////////////////////////////////////////////

  do {

    if (status == Solver::OUTPUT) {

      plot_count += 1;
      uCPU = *solver.u;
      uCPU_p = *solver.Xp;

      // Output data of gas-phase flow field //////////////////////////////////////////////////////////////////////////////////////////////////// 
      if (true) {  
    
        std::stringstream filename1;
        filename1 << solver.outputDirectory << "RESULT" << std::setw(6) << std::setfill('0') << solver.getOutputNumber() << ".plt";

        std::ofstream outFile;
        outFile.open(filename1.str().c_str());

        outFile.precision(8);
        outFile << "ZONE I = " << uCPU.activeNx() << " J = " << uCPU.activeNy() << " K = " << uCPU.activeNz() << " DATAPACKING = POINT" << std::endl;
        outFile << "DIMENSIONS " << uCPU.xMax() << " " << uCPU.yMax() << std::endl;

      //  for (int j = 0; j < uCPU.activeNy(); ++j) { 
          int j = 5;
          //  for (int k = 0; k < uCPU.activeNz(); ++k) { 
          int k = 5;
          for (int i = 0; i < uCPU.activeNx(); ++i) {
            outFile << std::fixed << uCPU.x(i) << " " << uCPU(i, j, k)[TG] << std::endl;
          }
     //   } 
     //   }
        outFile.close(); 
        
    }

    // Output data of solid particles //////////////////////////////////////////////////////////////////////////////////////////////////// 

      if (true) {  
    
        std::stringstream filename_p;
        filename_p << solver.outputDirectory << "Particles" << std::setw(6) << std::setfill('0') << solver.getOutputNumber() << ".plt";

        std::ofstream outFile_p;
        outFile_p.open(filename_p.str().c_str());

        outFile_p.precision(8);
        outFile_p << "VARIABLES = \"XP\", \"YP\", \"ZP\", \"TP\", \"Ignition Time\"" << std::endl;
        outFile_p << "NUMBER_OF_PARTICALES = " << uCPU_p.NP() << std::endl;

        for (int i = 0; i < uCPU_p.NP(); ++i) { 
          outFile_p << std::fixed << uCPU_p(i, XPOS) << " " <<  uCPU_p(i, YPOS) << " " <<  uCPU_p(i, ZPOS) << " " <<  uCPU_p(i, TP) <<  " " <<  uCPU_p(i, TAU) << std::endl;
        } 
        outFile_p.close(); 
        
    }

    // Output figures /////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    /*
      if (plot_count > plot_interval) {

        plot_count = 1;
        std::vector<ImageOutputs> outputs;

       //  outputs.push_back((ImageOutputs){"pressure", PRESSUREFOO, HUE, 10.0, 60.0});
        // outputs.push_back((ImageOutputs){"pressure2", PRESSUREFOO, HUE, 1.0, 50.0});
        outputs.push_back((ImageOutputs){"temperature", TEMPPLOT, GREYSCALE, Tf, 2.5*Tf});
     //   outputs.push_back((ImageOutputs){"frac", FRACTIONPLOT, HUE, 0.0, 1.0});
      //    outputs.push_back((ImageOutputs){"V_X", XVELOCITYPLOT, HUE, -V_CJ, V_CJ/2.0});
      //  outputs.push_back((ImageOutputs){"schlieren", SCHLIEREN, GREYSCALE, 0.0, 0.000001});
      //  outputs.push_back((ImageOutputs){"temperature", SOUNDSPEED, HUE, 1.0, 40.0});

        for (std::vector<ImageOutputs>::iterator iter = outputs.begin(); iter != outputs.end(); ++iter) {
          std::stringstream filename;
          filename << solver.outputDirectory << (*iter).prefix << std::setw(6) << std::setfill('0') << solver.getOutputNumber() << ".png";
          saveFrame(*solver.u, (*iter).plotVariable, (*iter).colourMode, filename.str().c_str(), (*iter).min, (*iter).max, & (*solver.fluxes)(0, 0, 0));
        }
      }
      */
    }
  } while ((status = solver.step()) != Solver::FINISHED && !halt);

  times(&endTimes);
  clock_gettime(CLOCK_REALTIME, &endClock);
  const double wallTime = (endClock.tv_sec - startClock.tv_sec) + (endClock.tv_nsec - startClock.tv_nsec) * 1e-9;

  std::cout << "CPU time, wall= " << std::setprecision(2) << std::fixed << wallTime << "s, user=" << (endTimes.tms_utime - endTimes.tms_utime) << "s, sys=" << (endTimes.tms_stime - endTimes.tms_stime) << "s.  Time for {fluxes=" << solver.getTimeFluxes() * 1e-3 << "s, sources=" << solver.getTimeSourcing() * 1e-3 << "s, reduction=" << solver.getTimeReducing() * 1e-3 << "s, adding=" << solver.getTimeAdding() * 1e-3 << "s }" << std::endl;
  return 0;
}


