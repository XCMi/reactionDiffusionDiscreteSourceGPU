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

const int NUMBER_VARIABLES = 2;
const int GHOST_CELLS = 1;
const int CONSERVATIVE_VARIABLES = 2;
const int NONCONSERVATIVE_VARIABLES = 0;
const int DUMMY_VARIABLES = 0;

enum ConservedVariables { TG, CG };
enum BoundaryConditions { ADIABATIC, ISOTHERMAL, PERIODIC };
enum Boundaries {LEFT, RIGHT, BOTTOM, TOP, BACK, FRONT};

#include "Timer.hpp"
#include "StridedArray.hpp"
#include "Grid3.hpp"
#include "Mesh3.hpp"

