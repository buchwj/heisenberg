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
 * Implementations of classes handling user input
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include "heisenberg.h"

InputFile::InputFile()
{

}

InputFile::InputFile(const char *filename)
{
  this->loadFile(filename);
}

InputFile::~InputFile()
{

}

void InputFile::loadFile(const char *filename)
{
  // set defaults
  setDefaults();

  // try to open input file
  FILE *f = fopen(filename, "r");
  if (!f) {
    printf("ERROR: could not open input file %s\n", filename);
    throw ERROR_IO;
  }

  // read all lines
  char peek;
  char label[20];
  char value[20];
  while (!feof(f)) {
    // check if comment
    peek = (char)fgetc(f);
    if (peek == '!') {
      // this line is a comment; skip to next line
      while (peek != '\n' && peek != EOF) peek = (char)fgetc(f);
      continue;
    }
    // not a comment, undo the peek
    ungetc(peek, f);

    // read line
    fscanf(f, "%s %s\n", label, value);

    if (strcmp(label, LABEL_LENGTH) == 0) {
      // this line specifies the edge length
      this->m_length = atoi(value);
      continue;
    }

    if (strcmp(label, LABEL_J) == 0) {
      // this line specifies the coupling constant
      this->m_j = atof(value);
      continue;
    }

    if (strcmp(label, "ext") == 0) {
      // this line specifies whether to apply an external field
      this->m_ext = false;
      if (value[0] == 't' || value[0] == 'T' || value[0] == '1') {
	this->m_ext = true;
      }
      continue;
    }

    if (strcmp(label, LABEL_MU) == 0) {
      // this line specifies the magnetic moment
      this->m_mu = atof(value);
      continue;
    }

    if (strcmp(label, LABEL_H_X) == 0) {
      // this line specifies the x component of the external field
      this->m_h_x = atof(value);
      continue;
    }

    if (strcmp(label, LABEL_H_Y) == 0) {
      // this line specifies the y component of the external field
      this->m_h_y = atof(value);
      continue;
    }

    if (strcmp(label, LABEL_H_Z) == 0) {
      // this line specifies the z component of the external field
      this->m_h_z = atof(value);
      continue;
    }

    if (strcmp(label, LABEL_SEED) == 0) {
      // this line specifies a fixed seed for the random number generator
      this->m_seed = atoll(value);
      continue;
    }

    if (strcmp(label, LABEL_PTAMP) == 0) {
      // this line specifies the "amplitude" of the random perturbations to the spin lattice
      this->m_ptamp = atof(value);
      continue;
    }

    if (strcmp(label, LABEL_RECFRQ) == 0) {
      // this line specifies the frequency with which to record the current state
      this->m_recfrq = atoi(value);
      continue;
    }

    // check if this is a thermostat command
    if (label[0] == 't' && label[1] == 'h') {
      parseThermostatCommand(label, value);
      continue;
    }

    // unrecognized input file option
    printf(" ERROR: Unrecognized input file option: %s\n", label);
    throw ERROR_INPUT;
  }
  fclose(f);
  f = NULL;
}

void InputFile::parseThermostatCommand(const char *label, const char *value)
{
  m_thncmd++;
  bool ok = false;
  if (strcmp(label, LABEL_THNSTEP) == 0) {
    // this line gives the length of this thermal command
    m_thccmd.steps = atoi(value);
    ok = true;
  }
  
  if (strcmp(label, LABEL_THSTART) == 0) {
    // this line gives the initial temperature of this thermal command
    m_thccmd.start = atof(value);
    ok = true;
  }

  if (strcmp(label, LABEL_THSLOPE) == 0) {
    // this line gives the slope of the temperature ramp
    m_thccmd.slope = atof(value);
    ok = true;
  }

  if (!ok) {
    // unrecognized command
    printf(" ERROR: Unrecognized input file option: %s\n", label);
    throw ERROR_INPUT;
  }

  if (m_thncmd == 3) {
    // full command loaded
    m_thermostat.addCommand(m_thccmd);
    m_thncmd = 0;
  }
}

void InputFile::printSummary()
{
  printf(" \n");
  printf(" Summary of current settings:\n");
  printf(" \tLength of cell = %d\n", m_length);
  printf(" \tExchange coupling = %f\n", m_j);
  printf(" \tExternal field is %s\n", m_ext ? "ON" : "OFF");
  if (m_ext) {
    printf(" \tMagnetic moment = %f\n", m_mu);
    printf(" \tExternal field = < %f, %f, %f >\n", m_h_x, m_h_y, m_h_z);
  }
  printf(" \tRandom seed = %d\n", m_seed);
  printf(" \tPerturbation amplitude = %f\n", m_ptamp);
  printf(" \tState recording frequency = %d\n", m_recfrq);
  printf(" \n");
  m_thermostat.printSummary();
  printf(" \n");
}

void InputFile::setDefaults()
{
  // initialize to default values
  m_length = DEFAULT_LENGTH;
  m_j = DEFAULT_J;
  m_ext = DEFAULT_EXT;
  m_mu = DEFAULT_MU;
  m_h_x = DEFAULT_H_X;
  m_h_y = DEFAULT_H_Y;
  m_h_z = DEFAULT_H_Z;
  m_ptamp = DEFAULT_PTAMP;

  // choose a default random seed
  m_seed = (long long int) time(NULL);

  // zero the thermal counter
  m_thncmd = 0;
}
