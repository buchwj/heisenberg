#
# PHYS 4960 Final Project
# GPU-Accelerated Classical Heisenberg Model
#
# James R. Buchwald
# Dept. of Chemistry and Chemical Biology
# Rensselaer Polytechnic Institute
# buchwj@rpi.edu
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

cmake_minimum_required(VERSION 2.6)

# project information
project(Heisenberg)
set(Heisenberg_VERSION_MAJOR 1)
set(Heisenberg_VERSION_MINOR 0)

# locate libraries
find_package(CUDA)
include_directories(${CMAKE_SOURCE_DIR}/cudpp/include)

# configuration file generation
configure_file(
  "${PROJECT_SOURCE_DIR}/config.h.in"
  "${PROJECT_BINARY_DIR}/config.h"
)
include_directories("${PROJECT_BINARY_DIR}")

# set architecture
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_35 -g -G)

# executables
CUDA_ADD_EXECUTABLE(heisenberg.x main.cpp input.cpp thermostat.cpp simulation.cpp simulation_kernels.cu)

# additional library dependencies
target_link_libraries(heisenberg.x ${CUDA_curand_LIBRARY} ${CMAKE_SOURCE_DIR}/cudpp/build/lib/libcudpp.a)
