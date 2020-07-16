/*
  Copyright © Computing Energetic - Calcul Energetique 2020
*/
#pragma once
#include <algorithm>
#include <iostream>
#include <ctime>
#include <cstdio>
#include <string.h>
#include <string>
#include "Vector.hpp"

double typedef real;

Vector<real, 2> typedef Vec;
const int NUMBER_VARIABLES_particle = 8;
// (XPOS, YPOS, ZPOS): x,y,z coordinates 
// TP: particle temperature
// CP: oxidizer concentration at particle surface
// MP: total mass of particle
// YF: Fuel mass fraction 
enum ConservedVariables_particle {XPOS, YPOS, ZPOS, TP, CP, MP, YF, TAU};
#include "Particles3.hpp"
#include "Cloud3.hpp"

