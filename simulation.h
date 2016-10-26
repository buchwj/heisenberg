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
 * Declarations of simulation classes
 */

#ifndef __HEISENBERG_SIMULATION_H__
#define __HEISENBERG_SIMULATION_H__

#define SITE_X(index,l) index / (l*l)
#define SITE_Y(index,l) (index % (l*l)) / l
#define SITE_Z(index,l) index % l
#define SITE_INDEX(x,y,z,l) x*l*l + y*l + z
// #define WRAP_COORD(x,l) x - (x / l) * l
#define WRAP_COORD(x,l) 

/* kernel launch functions - implemented in simulation_kernel.cu */

void launch_normalize_spins(double *spin_x, double *spin_y, double *spin_z, struct siminfo *info, int blocks, int threads);

void launch_calculate_internal_energies(double *spin_x, double *spin_y, double *spin_z, double *energy, struct siminfo *info, int blocks, int threads);

void launch_calculate_external_energies(double *spin_x, double *spin_y, double *spin_z, double *energy, struct siminfo *info, int blocks, int threads);

void launch_apply_perturbation(double *spin_x, double *spin_y, double *spin_z, double *perturb_x, 
				 double *perturb_y, double *perturb_z, double ptamp, struct siminfo *info, int blocks, int threads);

/* host code - implemented in simulation.cpp */

typedef struct siminfo
{
  double j;
  double mu;
  double h_x;
  double h_y;
  double h_z;
  unsigned int length;
  int threads_per_block;
  int num_blocks;
} *lpsiminfo;

class Simulation {
 public:
  Simulation();
  Simulation(InputFile*);
  virtual ~Simulation();

  // setters
  inline void setInput(InputFile *input) { m_input = input; }

  /** Initializes the simulation parameters */
  void initialize();

  /** Initialize CUDA */
  void initializeCuda();

  /** Allocates all buffers */
  void allocateBuffers();

  /** Frees all buffers */
  void freeBuffers();

  /** Initializes the random number generators */
  void initRNG();

  /** Cleans up the random number generators */
  void cleanupRNG();

  /** Randomly initializes the spins */
  void initializeSpins();

  /** Runs the Monte Carlo simulation */
  void run();

  /** Step the Monte Carlo simulation */
  void step();

 protected:
  // simulation parameters, etc.
  InputFile *m_input;

  // buffer size (must be calculated on-the-fly due to user-variable length)
  int m_bufsize;

  // host buffers
  double *m_h_spin_x;
  double *m_h_spin_y;
  double *m_h_spin_z;

  // device buffers
  struct siminfo *m_d_info;
  double *m_d_spin_x[2];    // two buffers
  double *m_d_spin_y[2];    // two buffers
  double *m_d_spin_z[2];    // two buffers
  double *m_d_energy;
  double *m_d_totale;
  double *m_d_perturb_x;
  double *m_d_perturb_y;
  double *m_d_perturb_z;
  
  // state parameters
  double m_temperature;
  double m_curenergy;
  double m_trialenergy;

  // timing
  clock_t m_t_perturb;
  clock_t m_t_inte;
  clock_t m_t_exte;
  clock_t m_t_reduce;

  // rng controls
  curandGenerator_t m_generator;

  // cudpp
  CUDPPHandle m_cudpph;
  CUDPPHandle m_reduceplan;

  // evolution controls
  int m_curstate;    // index to m_d_spin_* of current accepted state
  int m_curstep;     // step count

  // cuda block divisions
  int m_threads_per_block;
  int m_num_blocks;

  /** Load user input into device-side siminfo structure */
  void loadInput();

  /** Generate a trial state */
  void generateTrial();

  /** Determines the energy of the trial state */
  void calculateTrialEnergy();

  /** Exchanges the current and trial states */
  void exchangeStates();

  /** Determines whether to accept the trial state */
  bool shouldAcceptTrialState();

  /** Output results header */
  void outputHeader();

  /** Outputs results from a step */
  void outputResults(bool accepted);

  /** Save current state to disk */
  void saveState();

  /** Copies the current state from the GPU to the host */
  void retrieveState();

};

// number of buffers (used for displaying memory requirements)
#define NUM_HOST_BUFFERS 3
#define NUM_DEVICE_BUFFERS 10

#endif // __HEISENBERG_SIMULATION_H__
