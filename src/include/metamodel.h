/** RTDO: Closed loop neuron model fitting
 *  Copyright (C) 2019 Felix Kern <kernfel+github@gmail.com>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
**/


#ifndef METAMODEL_H
#define METAMODEL_H

#include <string>
#include <vector>
#include "types.h"
#include "../tinyxml2/tinyxml2.h"

class neuronModel;
class NNmodel;

class MetaModel
{
public:
    MetaModel(Project const& p, std::string file = "");

    neuronModel generate(NNmodel &m, std::vector<double> &fixedParamIni, std::vector<double> &variableIni) const;

    std::string supportCode() const;
    std::string populateStructs(std::string paramPre = "$(", std::string paramPost = ")",
                                std::string rundPre = "$(", std::string rundPost = ")") const;
    std::string extractState(std::string pre = "$(", std::string post = ")") const;
    std::string daqCode(int ordinal) const;

    std::string name() const; //!< Model name, no suffix

    std::vector<StateVariable> stateVariables;
    std::vector<AdjustableParam> adjustableParams;
    std::vector<Variable> currents;

    struct Current {
        std::string name;
        std::string popen;
        scalar gUnit = 0.02;
        std::vector<Variable*> gbar;
        Variable *E = nullptr;
        std::string optGroup;
        bool option;
    };
    std::vector<Current> currentDefs;
    StateVariable *V = nullptr;
    Variable *C = nullptr;

    int nNormalAdjustableParams;
    int nOptions = 0;

    /**
     * @brief get_detune_indices returns the indices into adjustableParams to be detuned for a given set of EE trajectories.
     * Trajectory starting points are indicated with a negative value, meaning no parameter needs to change. The value is -2
     * for the first starting points, until a full EE trajectory is completed (i.e., until all parameters and option switches are used once,
     * relevant e.g. for WavegenData::useBaseParameters), and -1 for later starting points.
     * @param trajLen Length of an individual EE trajectory, including the starting point
     * @param nTraj Number of trajectories
     * @return A vector of trajLen*nTraj ints indicating the adjustableParam to detune, or negative values for trajectory starting points.
     */
    std::vector<int> get_detune_indices(int trajLen, int nTraj) const;

    static void (*modelDef)(NNmodel&);

    static size_t numLibs; //!< Count the number of open libraries

    bool isUniversalLib = false;
    std::string save_state_condition = "";
    bool save_selectively = false;
    std::vector<std::string> save_selection;
    std::vector<TypedVariableBase*> singular_stim_vars, singular_clamp_vars, singular_rund_vars, singular_target_vars;

protected:
    const Project &project;
    std::string _name;
    std::vector<Variable> _params;

    bool isCurrent(const Variable &tmp) const;
    std::string resolveCode(const std::string &code) const;
    std::string structDeclarations() const;

    void readVariables(const tinyxml2::XMLElement *);
    void readParams(const tinyxml2::XMLElement *);
    void readAdjustableParams(const tinyxml2::XMLElement *);
    void readCurrents(const tinyxml2::XMLElement *);
    void readVoltage(const tinyxml2::XMLElement *);
    bool readCapacitance(const tinyxml2::XMLElement *);
};

// generateALL.cc's renamed main - see core/generateAllNoMain.cpp:
int generateAll(int argc, char *argv[]);

// The model definition that GeNN expects to be dropped into genn-buildmodel.sh:
inline void modelDefinition(NNmodel &n) { MetaModel::modelDef(n); }

// A copy of generateAll that correctly invokes the modified genNeuronKernel
// See core/generateAllNoMain.cpp
void generateCode(std::string path, const MetaModel &metamodel);

#endif // METAMODEL_H
