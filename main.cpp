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

#include <cstdio>

#include "heisenberg.h"

void output_program_info()
{
  printf("\n");
  printf(" *************************************\n");
  printf(" *************************************\n");
  printf(" ***                               ***\n");
  printf(" ***          HEISENBERG           ***\n");
  printf(" ***   GPU-Accelerated Magnetism   ***\n");
  printf(" ***   -------------------------   ***\n");
  printf(" ***       James R. Buchwald       ***\n");
  printf(" ***           PHYS 4960           ***\n");
  printf(" ***                               ***\n");
  printf(" *************************************\n");
  printf(" *************************************\n");
  printf("\n");
}

void print_usage(const char *exename)
{
  printf(" usage: %s inputfile\n", exename);
}

int main(int argc, char **argv)
{
  try {

    // check arguments
    if (argc != 2) {
      print_usage(argv[0]);
      return -1;
    }

    // output basic program information
    output_program_info();
  
    // process input file
    printf(" Loading input file from %s\n", argv[1]);
    InputFile *input = new InputFile(argv[1]);
    printf("  - Done.\n");

    // print summary of simulation parameters
    input->printSummary();

    // construct the simulation
    printf(" Initializing simulation\n");
    Simulation sim(input);
    sim.initialize();

    printf(" Initializing CUDA\n");
    sim.initializeCuda();
    printf("  - Done.\n");

    printf(" Allocating host and device memory buffers\n");
    sim.allocateBuffers();
    printf("  - Done.\n");

    printf(" Initializing random number generators\n");
    sim.initRNG();
    printf("  - Done.\n");

    printf(" Initializing all spins to random orientations\n");
    sim.initializeSpins();
    printf("  - Done.\n");

    // run the simulation
    printf(" \n ***\n Beginning simulation\n ***\n \n");
    sim.run();
    printf(" \n ***\n Simulation complete\n ***\n \n");

    // release resources
    printf(" Releasing random number generators\n");
    sim.cleanupRNG();
    printf("  - Done.\n");
    printf(" Releasing host and device memory buffers\n");
    sim.freeBuffers();
    printf("  - Done.\n");

    // release pointers
    delete input;
  }
  catch (int error) {
    printf(" *** Abnormal exit with error code %d\n", error);
    return error;
  }

  printf("\n HEISENBERG TERMINATED NORMALLY\n\n");

  return 0;
}
