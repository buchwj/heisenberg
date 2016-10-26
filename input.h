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
 * Definitions for user input handlers.
 */

#ifndef __HEISENBERG_INPUT_H__
#define __HEISENBERG_INPUT_H__

/**
 * Class that loads and parses input files
 */
class InputFile {
 public:
  InputFile();
  InputFile(const char *filename);
  virtual ~InputFile();

  /** Loads and parses an input file. */
  virtual void loadFile(const char *filename);

  /** Summarizes the currently-loaded settings. */
  void printSummary();

  /* accessors */
  inline unsigned int getLength() const { return m_length; }
  inline double getJ() const { return m_j; }
  inline double getMu() const { return m_mu; }
  inline double getHx() const { return m_h_x; }
  inline double getHy() const { return m_h_y; }
  inline double getHz() const { return m_h_z; }
  inline bool getExt() const { return m_ext; }
  inline long long int getSeed() const { return m_seed; }
  inline double getPerturbationAmplitude() const { return m_ptamp; }
  inline int getRecordFrequency() const { return m_recfrq; }

  inline Thermostat* getThermostat() { return &m_thermostat; }

 protected:
  unsigned int m_length;
  double m_j;
  bool m_ext;
  double m_mu;
  double m_h_x, m_h_y, m_h_z;
  long long int m_seed;
  double m_ptamp;
  int m_recfrq;

  Thermostat m_thermostat;
  thermal_command m_thccmd;
  int m_thncmd;

  void parseThermostatCommand(const char *label, const char *value);
  void setDefaults();

};

// allowed labels in input files
#define LABEL_LENGTH "length"
#define LABEL_J "j"
#define LABEL_EXT "ext"
#define LABEL_MU "mu"
#define LABEL_H_X "h_x"
#define LABEL_H_Y "h_y"
#define LABEL_H_Z "h_z"
#define LABEL_SEED "seed"
#define LABEL_PTAMP "ptamp"
#define LABEL_RECFRQ "recfrq"

#define LABEL_THNSTEP "thnstep"
#define LABEL_THSTART "thstart"
#define LABEL_THSLOPE "thslope"

// default values
#define DEFAULT_LENGTH 20
#define DEFAULT_J 1.0
#define DEFAULT_EXT false
#define DEFAULT_MU 1.0
#define DEFAULT_H_X 0.0
#define DEFAULT_H_Y 0.0
#define DEFAULT_H_Z 0.0
#define DEFAULT_PTAMP 0.01
#define DEFAULT_RECFRQ 100

#endif // __HEISENBERG_INPUT_H__
