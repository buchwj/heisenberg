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
 * Device code implementation
 */

#include "heisenberg.h"

/* device functions */

__device__ int wrap_coordinate(int k, int l)
{
  if (k < 0) return (k + l);
  if (k >= l) return (k - l);
  return k;
}

/* kernels */

__global__ void normalize_spins(double *spin_x, double *spin_y, double *spin_z, struct siminfo *info)
{
  // objective: normalize each spin vector to have a magnitude of 1.0
  // task distribution: 1 block, N threads
  int site = info->threads_per_block * blockIdx.x + threadIdx.x;
  double magnitude = sqrt(spin_x[site]*spin_x[site] + spin_y[site]*spin_y[site] + spin_z[site]*spin_z[site]);
  spin_x[site] /= magnitude;
  spin_y[site] /= magnitude;
  spin_z[site] /= magnitude;
}

__global__ void calculate_internal_energies(double *spin_x, double *spin_y, double *spin_z, double *energy, struct siminfo *info)
{
  // objective: calculate contributions to total energy from spin-spin interactions
  // task distribution: 1 block, N threads (one thread per site)
  // periodic boundary conditions are employed
  int site = info->threads_per_block * blockIdx.x + threadIdx.x;
  unsigned int length = info->length;

  // identify site
  int x = SITE_X(site, length);
  int y = SITE_Y(site, length);
  int z = SITE_Z(site, length);
  int neighbor;

  // prepare
  energy[site] = 0.0;

  /*
   * Known failure case:
   *  blockIdx.x = 12
   *  threadIdx.x = 192
   *  l = 100
   *  n = 1000000
   *  r = (1,24,80)
   *  site = 12480
   *  neighbor = -997520
   */

  // step through each nearest-neighbor interaction
  // (-,.,.)
  neighbor = SITE_INDEX(wrap_coordinate(x-1,length),y,z,length); // not working
  energy[site] += spin_x[site]*spin_x[neighbor] + spin_y[site]*spin_y[neighbor] + spin_z[site]*spin_z[neighbor]; // access violation (neighbor < 0)

  // (+,.,.)
  neighbor = SITE_INDEX(wrap_coordinate(x+1,length),y,z,length);
  energy[site] += spin_x[site]*spin_x[neighbor] + spin_y[site]*spin_y[neighbor] + spin_z[site]*spin_z[neighbor];

  // (.,-,.)
  neighbor = SITE_INDEX(x,wrap_coordinate(y-1,length),z,length);
  energy[site] += spin_x[site]*spin_x[neighbor] + spin_y[site]*spin_y[neighbor] + spin_z[site]*spin_z[neighbor];

  // (.,+,.)
  neighbor = SITE_INDEX(x,wrap_coordinate(y+1,length),z,length);
  energy[site] += spin_x[site]*spin_x[neighbor] + spin_y[site]*spin_y[neighbor] + spin_z[site]*spin_z[neighbor];

  // (.,.,-)
  neighbor = SITE_INDEX(x,y,wrap_coordinate(z-1,length),length);
  energy[site] += spin_x[site]*spin_x[neighbor] + spin_y[site]*spin_y[neighbor] + spin_z[site]*spin_z[neighbor];

  // (.,.,+)
  neighbor = SITE_INDEX(x,y,wrap_coordinate(z+1,length),length);
  energy[site] += spin_x[site]*spin_x[neighbor] + spin_y[site]*spin_y[neighbor] + spin_z[site]*spin_z[neighbor];

  // distribute the coupling constant
  energy[site] *= info->j;
}

/* MUST BE CALLED AFTER calculate_internal_energies */
__global__ void calculate_external_energies(double *spin_x, double *spin_y, double *spin_z, double *energy, struct siminfo *info)
{
  // objective: calculate contribution to total energy from spin-h interactions
  int site = info->threads_per_block * blockIdx.x + threadIdx.x;

  // do not zero energy, it already contains the internal contribution to energy
  
  // compute the interaction with the external field
  energy[site] += info->mu * (spin_x[site]*info->h_x + spin_y[site]*info->h_y + spin_z[site]*info->h_z);
}

__global__ void apply_perturbation(double *spin_x, double *spin_y, double *spin_z, double *perturb_x, 
				   double *perturb_y, double *perturb_z, double ptamp, struct siminfo *info)
{
  // objective: modulate the perturbation by the user-supplied amplitude ptamp, then apply it
  int site = info->threads_per_block * blockIdx.x + threadIdx.x;
  spin_x[site] += (perturb_x[site] - 0.5) * (ptamp * 2.0);
  spin_y[site] += (perturb_y[site] - 0.5) * (ptamp * 2.0);
  spin_z[site] += (perturb_z[site] - 0.5) * (ptamp * 2.0);
}

/* launchers (called from host) */
void launch_normalize_spins(double *spin_x, double *spin_y, double *spin_z, struct siminfo *info, int blocks, int threads)
{
  normalize_spins<<<blocks,threads>>>(spin_x,spin_y,spin_z,info);
}

void launch_calculate_internal_energies(double *spin_x, double *spin_y, double *spin_z, double *energy, struct siminfo *info, int blocks, int threads)
{
  calculate_internal_energies<<<blocks,threads>>>(spin_x,spin_y,spin_z,energy,info);
}

void launch_calculate_external_energies(double *spin_x, double *spin_y, double *spin_z, double *energy, struct siminfo *info, int blocks, int threads)
{
  calculate_external_energies<<<blocks,threads>>>(spin_x,spin_y,spin_z,energy,info);
}

void launch_apply_perturbation(double *spin_x, double *spin_y, double *spin_z, double *perturb_x, 
				 double *perturb_y, double *perturb_z, double ptamp, struct siminfo *info, int blocks, int threads)
{
  apply_perturbation<<<blocks,threads>>>(spin_x,spin_y,spin_z,perturb_x,perturb_y,perturb_z,ptamp,info);
}
