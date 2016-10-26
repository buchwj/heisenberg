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

/** Implementation of Thermostat class */

#include <stdio.h>
#include "heisenberg.h"

Thermostat::Thermostat()
{
  m_relstepnum = 0;
  m_temperature = 0.0;
}

Thermostat::~Thermostat()
{

}

// steps the thermostat forward once
void Thermostat::step()
{
  // get the current command
  thermal_command *cmd = &(m_commands.front());

  // have we taken enough steps to complete this command?
  if (m_relstepnum > cmd->steps) {
    // move to the next command
    nextCommand();

    // are there any more commands to process?
    if (isDone()) {
      // thermal program complete
      return;
    }
    // continue processing the next command
    cmd = &(m_commands.front());
    if (!cmd) {
      // abnormal case: did not recognize that we are done
      m_done = true;
      return;
    }
  }
  
  // execute next step of the thermal command
  m_temperature = cmd->start + (double)m_relstepnum * cmd->slope;
  m_relstepnum++;
}

// adds a command to the back of the queue
void Thermostat::addCommand(thermal_command cmd)
{
  m_commands.push_back(cmd);
}

// moves to the next command
void Thermostat::nextCommand()
{
  m_commands.pop_front();
  m_relstepnum = 0;
}

void Thermostat::printSummary()
{
  printf(" Summary of currently loaded thermal program:\n");
  printf(" \tNumber of commands = %d\n", m_commands.size());
  std::deque<thermal_command>::iterator itr = m_commands.begin();
  int i = 0;
  while (itr != m_commands.end()) {
    // advance iterator while getting handle to the command
    thermal_command *cmd = &(*itr++);

    // summarize command
    printf(" \tCommand %d\n", ++i);
    printf(" \t\tLength = %d steps\n", cmd->steps);
    printf(" \t\tInit temp = %f K\n", cmd->start);
    printf(" \t\tSlope = %f K/step\n", cmd->slope);
    printf(" \n");
  }
  printf(" End of thermal program.\n");
}
