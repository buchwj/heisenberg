/*
 * PHYS 4960 Final Project
 * GPU-Accelerated Classical Heisenberg Model
 *
 * James R. Buchwald
 * Dept. of Chemistry and Chemical Biology
 * Rensselaer Polytechnic Institute
 * buchwj@rpi.edu
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * Main include file for the project.
 */

#ifndef __HEISENBERG_H__
#define __HEISENBERG_H__

// type definitions
//typedef float hFloat;

// global headers
#include <cmath>
#include <string>
#include <deque>
#include <curand.h>
#include <cudpp.h>

// other local headers
#include "thermostat.h"
#include "input.h"
#include "simulation.h"

// error codes
#define ERROR_OK      0
#define ERROR_IO     -1
#define ERROR_MEM    -2
#define ERROR_CURAND -3
#define ERROR_CUDA   -4
#define ERROR_CUDPP  -5
#define ERROR_INPUT  -6

#endif // __HEISENBERG_H__
