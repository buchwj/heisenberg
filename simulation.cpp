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
 * Implementation of simulation (host code)
 */

#include <cstdio>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include "heisenberg.h"

/* Simulation class (host code) */

Simulation::Simulation()
{

}

Simulation::Simulation(InputFile *input)
{
  printf(" [debug] setting input\n");
  setInput(input);
}

Simulation::~Simulation()
{

}

void Simulation::initialize()
{
  m_h_spin_x = NULL;
  m_h_spin_y = NULL;
  m_h_spin_z = NULL;
  m_d_spin_x[0] = NULL;
  m_d_spin_x[1] = NULL;
  m_d_spin_y[0] = NULL;
  m_d_spin_y[1] = NULL;
  m_d_spin_z[0] = NULL;
  m_d_spin_z[1] = NULL;
  m_d_energy = NULL;
  m_d_totale = NULL;
  m_d_info = NULL;
  m_curstate = 0;
  m_temperature = 0.0;
  m_curstep = 0;
  m_curenergy = 0.0;
  m_trialenergy = 0.0;
  m_t_perturb = 0;
  m_t_inte = 0;
  m_t_exte = 0;
  m_t_reduce = 0;
}

void Simulation::initializeCuda()
{
  // identify cuda devices
  int devices;
  cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess) {
    // either no devices or no suitable drivers
    printf(" ERROR: cudaGetDeviceCount() failed: %d\n", err);
    throw ERROR_CUDA;
  }

  // take the first device
  err = cudaSetDevice(0);

  // get device properties
  cudaDeviceProp prop;
  err = cudaGetDeviceProperties(&prop, 0);
  if (err != cudaSuccess) {
    // error
    printf(" ERROR: could not get CUDA device properties: %d\n", err);
    throw ERROR_CUDA;
  }

  // summarize device properties to output
  printf(" Selected CUDA device 0.\n");
  printf("   Clock rate: %d kHz\n", prop.clockRate);
  printf("   Total device memory: %d MB\n", prop.totalGlobalMem / (1024*1024));
  printf("   Maximum threads per block: %d\n", prop.maxThreadsPerBlock);

  // subdivide blocks and threads
  m_threads_per_block = prop.maxThreadsPerBlock;
  int n = m_input->getLength(); // segfault
  n = n*n*n;
  m_num_blocks = n / m_threads_per_block + 1;

  // initialize cudpp
  cudppCreate(&m_cudpph);
}

void Simulation::allocateBuffers()
{
  // calculate size of buffers
  int length = m_input->getLength();
  int n = length * length * length;
  m_bufsize = n * sizeof(double);

  // display estimated memory requirements
  int gpumemreq = (NUM_DEVICE_BUFFERS * m_bufsize + sizeof(struct siminfo) + sizeof(double)) / 1024;
  printf(" \n");
  printf(" Summary of memory requirements (estimated):\n");
  printf("   Host memory to be dynamically allocated = %d kb\n", NUM_HOST_BUFFERS * m_bufsize / 1024);
  printf("   GPU memory to be dynamically allocated  = %d kb\n", gpumemreq);
  printf(" \n");

  // allocate host buffers
  m_h_spin_x = (double*) malloc(m_bufsize);
  if (!m_h_spin_x) {
    printf(" ERROR: Failed to allocate m_h_spin_x\n");
    throw ERROR_MEM;
  }

  m_h_spin_y = (double*) malloc(m_bufsize);
  if (!m_h_spin_y) {
    printf(" ERROR: Failed to allocate m_h_spin_y\n");
    throw ERROR_MEM;
  }

  m_h_spin_z = (double*) malloc(m_bufsize);
  if (!m_h_spin_z) {
    printf(" ERROR: Failed to allocate m_h_spin_z\n");
    throw ERROR_MEM;
  }

  // allocate device buffers
  cudaError_t err;
  err = cudaMalloc(&(m_d_spin_x[0]), m_bufsize);
  if (err != cudaSuccess) {
    printf(" ERROR: Failed to allocate m_d_spin_x[0]: %d\n", err);
    throw ERROR_MEM;
  }

  err = cudaMalloc(&(m_d_spin_x[1]), m_bufsize);
  if (err != cudaSuccess) {
    printf(" ERROR: Failed to allocate m_d_spin_x[1]: %d\n", err);
    throw ERROR_MEM;
  }

  err = cudaMalloc(&(m_d_spin_y[0]), m_bufsize);
  if (err != cudaSuccess) {
    printf(" ERROR: Failed to allocate m_d_spin_y[0]: %d\n", err);
    throw ERROR_MEM;
  }

  err = cudaMalloc(&(m_d_spin_y[1]), m_bufsize);
  if (err != cudaSuccess) {
    printf(" ERROR: Failed to allocate m_d_spin_y[1]: %d\n", err);
    throw ERROR_MEM;
  }

  err = cudaMalloc(&(m_d_spin_z[0]), m_bufsize);
  if (err != cudaSuccess) {
    printf(" ERROR: Failed to allocate m_d_spin_z[0]: %d\n", err);
    throw ERROR_MEM;
  }

  err = cudaMalloc(&(m_d_spin_z[1]), m_bufsize);
  if (err != cudaSuccess) {
    printf(" ERROR: Failed to allocate m_d_spin_z[1]: %d\n", err);
    throw ERROR_MEM;
  }

  err = cudaMalloc(&m_d_energy, m_bufsize);
  if (err != cudaSuccess) {
    printf(" ERROR: Failed to allocate m_d_energy: %d\n", err);
    throw ERROR_MEM;
  }

  // allocate device memory for simulation info
  err = cudaMalloc(&m_d_info, sizeof(struct siminfo));
  if (err != cudaSuccess) {
    printf(" ERROR: Failed to allocate m_d_info: %d\n", err);
    throw ERROR_MEM;
  }

  // load simulation info
  loadInput();

  // allocate the total energy variable (needed for the reduction)
  err = cudaMalloc(&m_d_totale, sizeof(double));
  if (err != cudaSuccess) {
    printf(" ERROR: Failed to allocate m_d_totale: %d\n", err);
    throw ERROR_MEM;
  }

  // allocate perturbation buffers
  err = cudaMalloc(&m_d_perturb_x, m_bufsize);
  if (err != cudaSuccess) {
    printf(" ERROR: Failed to allocate m_d_perturb_x: %d\n", err);
    throw ERROR_MEM;
  }

  err = cudaMalloc(&m_d_perturb_y, m_bufsize);
  if (err != cudaSuccess) {
    printf(" ERROR: Failed to allocate m_d_perturb_y: %d\n", err);
    throw ERROR_MEM;
  }

  err = cudaMalloc(&m_d_perturb_z, m_bufsize);
  if (err != cudaSuccess) {
    printf(" ERROR: Failed to allocate m_d_perturb_z: %d\n", err);
    throw ERROR_MEM;
  }

  // allocate the reduction plan
  CUDPPConfiguration config;
  config.algorithm = CUDPP_REDUCE;
  config.op = CUDPP_ADD;
  config.datatype = CUDPP_DOUBLE;
  config.options = 0;

  CUDPPResult cres = cudppPlan(m_cudpph, &m_reduceplan, config, n, 1, 0);
  if (cres != CUDPP_SUCCESS) {
    // error
    printf(" ERROR: Could not create reduction plan: %d\n", cres);
    throw ERROR_CUDPP;
  }
}

void Simulation::freeBuffers()
{
  // free reduction plan
  cudppDestroyPlan(m_reduceplan);

  // free perturbation buffers
  cudaFree(m_d_perturb_z);
  m_d_perturb_z = NULL;
  cudaFree(m_d_perturb_y);
  m_d_perturb_y = NULL;
  cudaFree(m_d_perturb_x);
  m_d_perturb_x = NULL;

  // free total energy variable
  cudaFree(m_d_totale);
  m_d_totale = NULL;

  // free simulation info
  cudaFree(m_d_info);
  m_d_info = NULL;

  // free device buffers
  cudaFree(m_d_spin_z[1]);
  m_d_spin_z[1] = NULL;
  cudaFree(m_d_spin_z[0]);
  m_d_spin_z[0] = NULL;
  cudaFree(m_d_spin_y[1]);
  m_d_spin_y[1] = NULL;
  cudaFree(m_d_spin_y[0]);
  m_d_spin_y[0] = NULL;
  cudaFree(m_d_spin_x[1]);
  m_d_spin_x[1] = NULL;
  cudaFree(m_d_spin_x[0]);
  m_d_spin_x[0] = NULL;

  // free host buffers
  free(m_h_spin_z);
  m_h_spin_z = NULL;
  free(m_h_spin_y);
  m_h_spin_y = NULL;
  free(m_h_spin_x);
  m_h_spin_x = NULL;
}

void Simulation::initRNG()
{
  // seed the host RNG
  srand((int) m_input->getSeed());
  
  // create the device RNG
  int status = curandCreateGenerator(&m_generator, CURAND_RNG_PSEUDO_MT19937);
  if (status != CURAND_STATUS_SUCCESS) {
    // error
    printf(" ERROR: could not create GPU random number generator: %d\n", status);
    throw ERROR_CURAND;
  }

  // seed the device RNG
  status = curandSetPseudoRandomGeneratorSeed(m_generator, m_input->getSeed());
  if (status != CURAND_STATUS_SUCCESS) {
    // error
    printf(" ERROR: could not seed the GPU random number generator: %d\n", status);
    throw ERROR_CURAND;
  }
}

void Simulation::cleanupRNG()
{
  // destroy the curand generator
  int status = curandDestroyGenerator(m_generator);
  if (status != CURAND_STATUS_SUCCESS) {
    // error
    printf(" ERROR: could not release the GPU random number generator: %d\n", status);
    throw ERROR_CURAND;
  }
}

void Simulation::initializeSpins()
{
  // objective: distribute initial spins subject to the constraint Sx^2+Sy^2+Sz^2 = 1.0
  unsigned int len = m_input->getLength();
  unsigned int n = len*len*len;

  // first step: distribute random spin components on the interval [0,1]
  int status = curandGenerateUniformDouble(m_generator, m_d_spin_x[0], n);
  if (status != CURAND_STATUS_SUCCESS) {
    // error
    printf(" ERROR: could not generate spin x components: %d\n", status);
    throw ERROR_CURAND;
  }

  status = curandGenerateUniformDouble(m_generator, m_d_spin_y[0], n);
  if (status != CURAND_STATUS_SUCCESS) {
    // error
    printf(" ERROR: could not generate spin y components: %d\n", status);
    throw ERROR_CURAND;
  }

  status = curandGenerateUniformDouble(m_generator, m_d_spin_z[0], n);
  if (status != CURAND_STATUS_SUCCESS) {
    // error
    printf(" ERROR: could not generate spin z components: %d\n", status);
  }

  // second step: normalize the spins
  launch_normalize_spins(m_d_spin_x[0], m_d_spin_y[0], m_d_spin_z[0], m_d_info, m_num_blocks, m_threads_per_block);

  // third step: copy the initial state into the trial buffer
  cudaError_t err = cudaMemcpy(m_d_spin_x[1], m_d_spin_x[0], m_bufsize, cudaMemcpyDeviceToDevice);

  // fourth step: calculate total energy of initial state
  m_curstate = 1;
  calculateTrialEnergy();
  m_curstate = 0;
  m_curenergy = m_trialenergy;
}

void Simulation::loadInput()
{
  // create and fill a local copy of the siminfo structure
  struct siminfo info;
  info.j = m_input->getJ();
  info.mu = m_input->getMu();
  info.h_x = m_input->getHx();
  info.h_y = m_input->getHy();
  info.h_z = m_input->getHz();
  info.length = m_input->getLength();
  info.threads_per_block = m_threads_per_block;
  info.num_blocks = m_num_blocks;

  // copy local structure to the GPU
  cudaError_t err = cudaMemcpy(m_d_info, &info, sizeof(struct siminfo), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    // error
    printf(" ERROR: could not copy simulation info to GPU: %d\n", err);
    throw ERROR_CUDA;
  }
}

void Simulation::generateTrial()
{
  /*
   * Four possible ways to generate a trial:
   *   1) Generate two random angles of rotation for each spin, then apply a rotation matrix
   *   2) Randomly perturb the spin components, then normalize the perturbed state
   *   3) Apply method (1) to a single spin
   *   4) Apply method (2) to a single spin
   *
   * The first method requires:
   *   - Generation of 2N random doubles in a uniform distribution
   *   - Evaluation of 4N trigonometric functions and 4N multiplications to generate the rotation matrices
   *   - 8N multiplications and 5N additions to apply the rotation matrices
   *
   * The second method requires:
   *   - Generation of 3N random doubles in a uniform distribution
   *   - 3N addition operations to apply the perturbation
   *   - 3N multiplications to compute the perturbed norm-squared
   *   - 3N divisions to normalize the spin vectors
   *
   * The third method requires:
   *   - Generation of 2 random doubles in a uniform distribution
   *   - Evaluation of 4 trigonometric functions and 4 multiplications to generate the rotation matrix
   *   - 8 multiplications and 5 additions to apply the rotation matrix
   *
   * The fourth method requires:
   *   - Generation of 3 random doubles in a uniform distribution
   *   - 3 addition operations ot apply the perturbation
   *   - 3 multiplications to compute the perturbed norm-squared
   *   - 3 divisions to normalize the spin vector
   *
   * It is clear that method (1) scales quite poorly relative to method (2), and it is reasonable to expect
   * that (2) and (4) are faster than (1) and (3), respectively.  Methods (2) and (4) have a further advantage
   * in that the extra normalization at each trial ensures that the normalization condition does not "drift"
   * over hundreds of thousands of trials due to roundoff error; that is, per-trial normalization avoids the
   * possibility that the rotation matrices become nonunitary over many trials due to roundoff error.
   *
   * For these reasons, we implement method (2) here.  The "amplitude" of the random perturbation is supplied
   * by the user input file.
   */
  
  // copy the current state to the trial state
  int ts = 1 - m_curstate;
  cudaError_t err = cudaMemcpy(m_d_spin_x[ts], m_d_spin_x[m_curstate], m_bufsize, cudaMemcpyDeviceToDevice);
  if (err != cudaSuccess) {
    // error
    printf(" ERROR: Could not initialize x components of trial state: %d\n", err);
    throw ERROR_CUDA;
  }

  err = cudaMemcpy(m_d_spin_y[ts], m_d_spin_y[m_curstate], m_bufsize, cudaMemcpyDeviceToDevice);
  if (err != cudaSuccess) {
    // error
    printf(" ERROR: Could not initialize y components of trial state: %d\n", err);
    throw ERROR_CUDA;
  }

  err = cudaMemcpy(m_d_spin_z[ts], m_d_spin_z[m_curstate], m_bufsize, cudaMemcpyDeviceToDevice);
  if (err != cudaSuccess) {
    // error
    printf(" ERROR: Could not initialize z components of trial state: %d\n", err);
    throw ERROR_CUDA;
  }

  // generate the random perturbation
  unsigned int n = m_bufsize / sizeof(double);
  int status = curandGenerateUniformDouble(m_generator, m_d_perturb_x, n);
  if (status != CURAND_STATUS_SUCCESS) {
    // error
    printf(" ERROR: Could not populate m_d_perturb_x: %d\n", status);
    throw ERROR_CURAND;
  }

  status = curandGenerateUniformDouble(m_generator, m_d_perturb_y, n);
  if (status != CURAND_STATUS_SUCCESS) {
    // error
    printf(" ERROR: Could not populate m_d_perturb_y: %d\n", status);
    throw ERROR_CURAND;
  }

  status = curandGenerateUniformDouble(m_generator, m_d_perturb_z, n);
  if (status != CURAND_STATUS_SUCCESS) {
    // error
    printf(" ERROR: Could not populate m_d_perturb_z: %d\n", status);
    throw ERROR_CURAND;
  }

  // apply the random perturbation and normalize the trial state
  launch_apply_perturbation(m_d_spin_x[ts], m_d_spin_y[ts], m_d_spin_z[ts], 
			    m_d_perturb_x, m_d_perturb_y, m_d_perturb_z,
			    m_input->getPerturbationAmplitude(), m_d_info,
			    m_num_blocks, m_threads_per_block);

  launch_normalize_spins(m_d_spin_x[ts], m_d_spin_y[ts], m_d_spin_z[ts], m_d_info, m_num_blocks, m_threads_per_block);
}

void Simulation::calculateTrialEnergy()
{
  // Calculate the single-site energies of the trial configuration
  unsigned int n = m_bufsize / sizeof(double);
  int ts = 1 - m_curstate; // ts = 2009007201; m_curstate not initialized?
  clock_t t0 = clock();
  launch_calculate_internal_energies(m_d_spin_x[ts], m_d_spin_y[ts], m_d_spin_z[ts], m_d_energy, m_d_info, m_num_blocks, m_threads_per_block);
  cudaDeviceSynchronize();
  clock_t tf = clock();
  m_t_inte += (tf - t0);
  if (m_input->getExt()) {
    // calculate interaction with an external field as well
    t0 = clock();
    launch_calculate_external_energies(m_d_spin_x[ts], m_d_spin_y[ts], m_d_spin_z[ts], m_d_energy, m_d_info, m_num_blocks, m_threads_per_block);
    cudaDeviceSynchronize();
    tf = clock();
    m_t_exte += (tf - t0);
  }

  // Reduce the energy vector to a total energy
  t0 = clock();
  CUDPPResult res = cudppReduce(m_reduceplan, m_d_totale, m_d_energy, n);
  cudaDeviceSynchronize();
  tf = clock();
  m_t_reduce += (tf - t0);
  if (res != CUDPP_SUCCESS) {
    // error
    printf(" ERROR: could not reduce energies: %d\n", res);
    throw ERROR_CUDPP;
  }

  // Fetch the total energy
  cudaError_t err = cudaMemcpy(&m_trialenergy, m_d_totale, sizeof(double), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    // error
    printf(" ERROR: could not fetch total energy: %d\n", res);
    throw ERROR_CUDA;
  }
}

void Simulation::exchangeStates()
{
  // swap total energies
  double tmp = m_curenergy;
  m_curenergy = m_trialenergy;
  m_trialenergy = tmp;

  // flip trial flag
  m_curstate = 1 - m_curstate;
}

bool Simulation::shouldAcceptTrialState()
{
  // if trial energy is less than current energy, always accept
  if (m_trialenergy < m_curenergy) return true;

  // otherwise weight the acceptance by a boltzmann distribution
  // (see Frenkel & Smit, pg 30)
  double key = (double)(rand()) / (double)(RAND_MAX);
  double delta = m_trialenergy - m_curenergy;
  double acc = exp(-1.0 * delta / m_temperature);
  return (key < acc);
}

void Simulation::step()
{
  // increment step counter
  m_curstep++;

  // update temperature
  Thermostat *therm = m_input->getThermostat();
  therm->step();
  if (therm->isDone()) {
    // simulation complete - this will be checked again outside of the step() call
    return;
  }
  m_temperature = therm->getTemperature();

  // generate a nearby trial state
  clock_t t0 = clock();
  generateTrial();
  cudaDeviceSynchronize();
  clock_t tf = clock();
  m_t_perturb += (tf - t0);

  // evaluate the energy of the trial state
  calculateTrialEnergy();

  // decide whether to keep the trial state
  bool accept = shouldAcceptTrialState();

  // output results
  outputResults(accept);

  // exchange states if trial accepted
  if (accept) {
    exchangeStates();
  }

  // possibly save state to disk
  if (m_curstep % m_input->getRecordFrequency() == 0) {
    // save state to disk
    saveState();
  }
}

void Simulation::outputResults(bool accepted)
{
  printf(" %d\t%.4f\t%.8f\t%.8f\t%s\n",  \
	 m_curstep,                      \
	 m_temperature,                  \
	 m_curenergy,                    \
	 m_trialenergy,                  \
	 accepted ? "ACCEPT" : "REJECT");
}

void Simulation::outputHeader()
{
  printf(" Step\tTemp\tCurE\tTriE\tAcc?\n");
}

void Simulation::saveState()
{
  // generate filename for state
  char filename[128];
  sprintf(filename, "step%d.dat", m_curstep);
  //  printf(" Saving current state at step %d to %s...\n", m_curstep, filename);

  // open file
  FILE *f = NULL;
  f = fopen(filename, "w");
  if (!f) {
    // couldn't open file
    printf(" ERROR: Could not open %s for writing.\n", filename);
    throw ERROR_IO;
  }

  // fetch current state from the GPU
  retrieveState();

  // output current state to file
  int l = m_input->getLength();
  int n = m_bufsize / sizeof(double);
  for (int i = 0; i < n; ++i) {
    fprintf(f,                                 \
	    "%d\t%d\t%d\t%.8f\t%.8f\t%.8f\n",  \
	    SITE_X(i,l),                       \
	    SITE_Y(i,l),                       \
	    SITE_Z(i,l),                       \
	    m_h_spin_x[i],                     \
	    m_h_spin_y[i],                     \
	    m_h_spin_z[i]);
  }

  // close file
  fclose(f);
  f = NULL;
}

void Simulation::retrieveState()
{
  cudaError_t err;

  err = cudaMemcpy(m_h_spin_x, m_d_spin_x[m_curstate], m_bufsize, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    printf(" ERROR: Could not retrieve x vectors of current state: %d\n", err);
    throw ERROR_CUDA;
  }

  err = cudaMemcpy(m_h_spin_y, m_d_spin_y[m_curstate], m_bufsize, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    printf(" ERROR: Could not retrieve y vectors of current state: %d\n", err);
    throw ERROR_CUDA;
  }

  err = cudaMemcpy(m_h_spin_z, m_d_spin_z[m_curstate], m_bufsize, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    printf(" ERROR: Could not retrieve z vectors of current state: %d\n", err);
    throw ERROR_CUDA;
  }
}

void Simulation::run()
{
  // output simulation header and initial state energy
  outputHeader();
  outputResults(true);

  bool done = false;
  while (!done) {
    // take a step
    step();

    // has the thermal program ended?
    Thermostat *therm = m_input->getThermostat();
    if (therm->isDone()) {
      // thermal program ended
      done = true;
      continue;
    }
  }

  // report timing info
  printf(" Summary of program timing:\n");
  printf("   Time to generate trial states       = %f s\n", (float)m_t_perturb / (float)CLOCKS_PER_SEC);
  printf("   Time to calculate internal energies = %f s\n", (float)m_t_inte / (float)CLOCKS_PER_SEC);
  printf("   Time to calculate external energies = %f s\n", (float)m_t_exte / (float)CLOCKS_PER_SEC);
  printf("   Time to sum energies by reduction   = %f s\n", (float)m_t_reduce / (float)CLOCKS_PER_SEC);
  printf("   Total time elapsed                  = %f s\n", (float)(m_t_perturb+m_t_inte+m_t_exte+m_t_reduce) / (float)CLOCKS_PER_SEC);
  printf(" \n");
}
