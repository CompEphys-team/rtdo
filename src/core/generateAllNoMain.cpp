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


// GeNN's generateALL.cc has an int main(int, char*[]), because it's meant to be compiled and run separately.
// This just allows us to compile it into the main project without conflict.
#define main generateAll

// MODEL is included in generateALL.cc; in standard GeNN use, this is a code file with the model definition,
// passed in through genn-buildmodel.sh
#define MODEL "metamodel.h"

#include "lib/src/generateALL.cc"

// To further confuse things (but keep the rewriting to a minimum):
// generateCode is called by the libraries to invoke the code generation step of GeNN.
// Previously, the libraries used to call GeNN::generateALL.cc's int main(int,char*[]) using the redirection above.
// Now, this function essentially replicates GeNN::main, but with a custom genNeuronKernel function.
static const MetaModel *pModel;
void generateCode(std::string path, const MetaModel &metamodel)
{
    pModel =& metamodel;
#ifdef DEBUG
    GENN_PREFERENCES::optimizeCode = false;
    GENN_PREFERENCES::debugCode = true;
#endif // DEBUG

#ifndef CPU_ONLY
    CHECK_CUDA_ERRORS(cudaGetDeviceCount(&deviceCount));
    deviceProp = new cudaDeviceProp[deviceCount];
    for (int device = 0; device < deviceCount; device++) {
        CHECK_CUDA_ERRORS(cudaSetDevice(device));
        CHECK_CUDA_ERRORS(cudaGetDeviceProperties(&(deviceProp[device]), device));
    }
#endif // CPU_ONLY

    NNmodel *model = new NNmodel();
    modelDefinition(*model);

#ifndef CPU_ONLY
    chooseDevice(*model, path);
#endif // CPU_ONLY
    generate_model_runner(*model, path);
}

// This is a stripped down version of GeNN::genNeuronKernel (see genn/lib/src/generateKernels.cc, which is no longer included in RTDO source)
// The key difference is the removal of redundant local variables which, in RTDO context, were then simply
// copied to the Parameters and State structs, as well as omitted storage of Parameters members back to global memory.
// Additionally, synapse functionality and related synchronisations are removed, as RTDO does not require them.
void genNeuronKernel(NNmodel &model, string &path)
{
    /// Hacking into definitions.h to add types.h inclusion and launch parameters
    std::string def_name_old = path + "/" + model.name + "_CODE/definitions.h";
    std::string def_name_new = def_name_old + ".new";
    ifstream definitions_old(def_name_old);
    ofstream def(def_name_new);
    def << "#include \"types.h\"\n";
    def << "#ifndef DEFINITIONS_H_LAUNCHPARAMS\n"
                    << "#define DEFINITIONS_H_LAUNCHPARAMS\n";
    {
        // Copied from generateRunner.cc:2293ff
        unsigned int neuronGridSz = model.padSumNeuronN[model.neuronGrpN - 1];
        neuronGridSz = ceil((float) neuronGridSz / neuronBlkSz);
        def << "dim3 nThreads_calcNeurons(" << neuronBlkSz << ", 1);" << ENDL;
        if (int(neuronGridSz) < deviceProp[theDevice].maxGridSize[1]) {
           def << "dim3 nGrid_calcNeurons(" << neuronGridSz << ", 1);" << ENDL;
        } else {
            int sqGridSize = ceil((float) sqrt((float) neuronGridSz));
            def << "dim3 nGrid(" << sqGridSize << ","<< sqGridSize <<");" << ENDL;
        }

        // forward declare calcNeurons (from below)
        def << "extern \"C\" __global__ void calcNeurons(";
        for (int i= 0, l= model.neuronKernelParameters.size(); i < l; i++) {
            def << model.neuronKernelParameterTypes[i] << " " << model.neuronKernelParameters[i] << ", ";
        }
        def << "int iT);" << ENDL;

    }
    def << "#endif\n";
    def << definitions_old.rdbuf();
    def.close();
    definitions_old.close();
    std::remove(def_name_old.c_str());
    std::rename(def_name_new.c_str(), def_name_old.c_str());
    /// Done.

    string name, localID;
    unsigned int nt;
    ofstream os;

    name = path + toString("/") + model.name + toString("_CODE/neuronKrnl.cc");
    os.open(name.c_str());

    // write header content
    writeHeader(os);
    os << ENDL;

    // compiler/include control (include once)
    os << "#ifndef _" << model.name << "_neuronKrnl_cc" << ENDL;
    os << "#define _" << model.name << "_neuronKrnl_cc" << ENDL;
    os << ENDL;

    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << ENDL;
    os << "/*! \\file neuronKrnl.cc" << ENDL << ENDL;
    os << "\\brief File generated from RTDO/GeNN for the model " << model.name << " containing the neuron kernel function." << ENDL;
    os << "*/" << ENDL;
    os << "//-------------------------------------------------------------------------" << ENDL << ENDL;

    os << "// include the support codes provided by the user for neuron or synaptic models" << ENDL;
    os << "#include \"support_code.h\"" << ENDL << ENDL;

    // kernel header
    os << "extern \"C\" __global__ void calcNeurons(";
    for (int i= 0, l= model.neuronKernelParameters.size(); i < l; i++) {
        os << model.neuronKernelParameterTypes[i] << " " << model.neuronKernelParameters[i] << ", ";
    }
    os << "int iT)" << ENDL;
    os << OB(5);

    // kernel code
    int neuronGridSz = model.padSumNeuronN[model.neuronGrpN - 1];
    neuronGridSz = neuronGridSz / neuronBlkSz;
    if (neuronGridSz < deviceProp[theDevice].maxGridSize[1]) {
        os << "unsigned int id = " << neuronBlkSz << " * blockIdx.x + threadIdx.x;" << ENDL;
    }
    else {
        os << "unsigned int id = " << neuronBlkSz << " * (blockIdx.x * " << ceil(sqrt((float) neuronGridSz)) << " + blockIdx.y) + threadIdx.x;" << ENDL;
    }
    os << ENDL;


    for (size_t i = 0; i < model.neuronGrpN; i++) {
        nt = model.neuronType[i];

        os << "// neuron group " << model.neuronName[i] << ENDL;
        if (i == 0) {
            os << "if (id < " << model.padSumNeuronN[i] << ")" << OB(10);
            localID = string("id");
        }
        else {
            os << "if ((id >= " << model.padSumNeuronN[i - 1] << ") && (id < " << model.padSumNeuronN[i] << "))" << OB(10);
            os << "unsigned int lid = id - " << model.padSumNeuronN[i - 1] << ";" << ENDL;
            localID = string("lid");
        }

        os << "// only do this for existing neurons" << ENDL;
        os << "if (" << localID << " < " << model.neuronN[i] << ")" << OB(20);

        os << "// pull neuron variables in a coalesced access" << ENDL;


/// *********************** RTDO edit: reading state & params ***********************************
        std::string ass = std::string("assignment") + model.neuronName[i];
        os << std::endl << pModel->populateStructs("dd_", model.neuronName[i] + "[(" + ass + " & ASSIGNMENT_SUBSET_MUX) ? (" + localID + " % cl_blocksizeUNI) : " + localID + "]", "", model.neuronName[i]) << ENDL;

        for (size_t k = 0; k < nModels[nt].varNames.size(); k++) {
            bool include = true;
            for ( const StateVariable &var : pModel->stateVariables )
                if ( var.name == nModels[nt].varNames[k] )
                    include = false;
            for ( const AdjustableParam &param : pModel->adjustableParams )
                if ( param.name == nModels[nt].varNames[k] )
                    include = false;
            if ( pModel->isUniversalLib ) {
                for ( const TypedVariableBase *var : pModel->singular_stim_vars )
                    if ( var->name == nModels[nt].varNames[k] )
                        include = false;
                for ( const TypedVariableBase *var : pModel->singular_rund_vars )
                    if ( var->name == nModels[nt].varNames[k] )
                        include = false;
                for ( const TypedVariableBase *var : pModel->singular_clamp_vars )
                    if ( var->name == nModels[nt].varNames[k] )
                        include = false;
                for ( const TypedVariableBase *var : pModel->singular_target_vars )
                    if ( var->name == nModels[nt].varNames[k] )
                        include = false;
            }
            if ( include ) {
                os << nModels[nt].varTypes[k] << " l" << nModels[nt].varNames[k] << " = dd_";
                os << nModels[nt].varNames[k] << model.neuronName[i] << "[" << localID << "];" << ENDL;
            }
        }
        if ( pModel->isUniversalLib ) {
            for ( const TypedVariableBase *var : pModel->singular_stim_vars )
                os << var->type << " l" << var->name << " = (" << ass << " & ASSIGNMENT_SINGULAR_STIM) ? singular_" << var->name
                   << " : dd_" << var->name << model.neuronName[i] << "["
                   << "(" << ass << " & ASSIGNMENT_SUBSET_MUX) ? " << localID << " / cl_blocksizeUNI : " << localID << "];" << ENDL;
            for ( const TypedVariableBase *var : pModel->singular_rund_vars )
                os << var->type << " l" << var->name << " = (" << ass << " & ASSIGNMENT_SINGULAR_RUND) ? singular_" << var->name
                   << " : dd_" << var->name << model.neuronName[i] << "["
                   << "(" << ass << " & ASSIGNMENT_SUBSET_MUX) ? " << localID << " % cl_blocksizeUNI : " << localID << "];" << ENDL;
            for ( const TypedVariableBase *var : pModel->singular_target_vars )
                os << var->type << " l" << var->name << " = (" << ass << " & ASSIGNMENT_SINGULAR_TARGET) ? singular_" << var->name
                   << " : dd_" << var->name << model.neuronName[i] << "["
                   << "(" << ass << " & ASSIGNMENT_SUBSET_MUX) ? " << localID << " / cl_blocksizeUNI : " << localID << "];" << ENDL;
        }
        os << ENDL;
/// ************************** RTDO edit ends **************************************************


        os << "// calculate membrane potential" << ENDL;
        string sCode = nModels[nt].simCode;
        substitute(sCode, tS("$(id)"), localID);
        substitute(sCode, tS("$(t)"), tS("t"));
        name_substitutions(sCode, tS("l"), nModels[nt].varNames, tS(""));
        value_substitutions(sCode, nModels[nt].pNames, model.neuronPara[i]);
        value_substitutions(sCode, nModels[nt].dpNames, model.dnp[i]);
        name_substitutions(sCode, tS(""), nModels[nt].extraGlobalNeuronKernelParameters, model.neuronName[i]);
        substitute(sCode, tS("$(Isyn)"), tS("Isyn"));
        substitute(sCode, tS("$(sT)"), tS("lsT"));
        sCode= ensureFtype(sCode, model.ftype);
        checkUnreplacedVariables(sCode,tS("neuron simCode"));

        if (nModels[nt].supportCode != tS("")) {
            os << OB(29) << " using namespace " << model.neuronName[i] << "_neuron;" << ENDL;
        }
        os << sCode << ENDL;
        if (nModels[nt].supportCode != tS("")) {
            os << CB(29) << " // namespace bracket closed" << endl;
        }


/// *********************** RTDO edit: storing state *******************************************
        if ( !pModel->save_state_condition.empty() ) {
            std::string condCode = pModel->save_state_condition;
            substitute(condCode, tS("$(id)"), localID);
            name_substitutions(condCode, tS("l"), nModels[nt].varNames, tS(""));
            name_substitutions(condCode, tS(""), nModels[nt].extraGlobalNeuronKernelParameters, model.neuronName[i]);
            checkUnreplacedVariables(condCode,tS("neuron simCode"));
            os << ENDL << "if ( " << condCode << " )" << OB(30);
        }
        os << std::endl << pModel->extractState("dd_", model.neuronName[i] + "[" + localID + "]") << ENDL;
        if ( !pModel->save_state_condition.empty() )
            os << CB(30);

        for (size_t k = 0; k < nModels[nt].varNames.size(); k++) {
            bool include;
            if ( pModel->save_selectively ) {
                include = false;
                for ( const std::string &varname : pModel->save_selection )
                    if ( varname == nModels[nt].varNames[k] )
                        include = true;
            } else {
                include = true;
                for ( const StateVariable &var : pModel->stateVariables )
                    if ( var.name == nModels[nt].varNames[k] )
                        include = false;
                for ( const AdjustableParam &param : pModel->adjustableParams )
                    if ( param.name == nModels[nt].varNames[k] )
                        include = false;
            }
            if ( include ) {
                os << "dd_" << nModels[nt].varNames[k] << model.neuronName[i] << "[" << localID << "] = l" << nModels[nt].varNames[k] << ";" << ENDL;
            }
        }
/// ************************** RTDO edit ends **************************************************


        os << CB(20);
        os << CB(10); // end if (id < model.padSumNeuronN[i] )
        os << ENDL;
    }
    os << CB(5) << ENDL; // end of neuron kernel

    os << "#endif" << ENDL;
    os.close();
}

void genSynapseKernel(NNmodel &, string &) {}
