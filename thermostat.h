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

/** Declaration of thermostat classes */

#ifndef __HEISENBERG_THERMOSTAT_H__
#define __HEISENBERG_THERMOSTAT_H__

typedef struct thermal_command
{
  double slope;
  double start;
  int steps;
} *lpthermal_command;

class Thermostat
{
 public:
  Thermostat();
  virtual ~Thermostat();

  /** gets the current temperature */
  inline double getTemperature() const { return m_temperature; }

  /** checks whether the thermal program is complete */
  inline bool isDone() const { return m_done; }

  /** steps the thermostat forward once */
  void step();

  /** adds a thermostat command */
  void addCommand(thermal_command cmd);

  /** prints a summary of the thermal program */
  void printSummary();

 protected:
  double m_temperature;
  bool m_done;

  std::deque<thermal_command> m_commands; // current command sits in front of queue
  int m_relstepnum; // steps since start of current command

  /** moves to the next command */
  void nextCommand();

};

#endif // __HEISENBERG_THERMOSTAT_H__
