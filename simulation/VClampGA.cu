/*--------------------------------------------------------------------------
Author: Thomas Nowotny

Institute: Informatics
University of Sussex
Brighton BN1 9QJ, UK

email to:  t.nowotny@sussex.ac.uk

initial version: 2014-06-26

--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file VClampGA.cu

\brief Main entry point for the GeNN project demonstrating realtime fitting of a neuron with a GA running mostly on the GPU.
*/
//--------------------------------------------------------------------------
#define _V_CLAMP
#ifdef _V_CLAMP
#define _CRTDBG_MAP_ALLOC
#include "VClampGA.h"
#include "realtimeenvironment.h"
#include "config.h"

//--------------------------------------------------------------------------
/*! \brief This function is the entry point for running the project
*/
//--------------------------------------------------------------------------

inline bool file_exists( const std::string& name )
{
	if (FILE *file = fopen( name.c_str(), "r" )) {
		fclose( file );
		return true;
	}
	else {
		return false;
	}
}

extern "C" int vclamp(conf::Config *cfg, bool *stopFlag, backlog::BacklogVirtual *logger) {
    RealtimeEnvironment &env = RealtimeEnvironment::env();

	//-----------------------------------------------------------------
    // read the relevant stimulus patterns
    vector<vector<double> > pperturb;
    vector<vector<double> > sigadjust;
	vector<inputSpec> stims;
    inputSpec I;
    ifstream is(cfg->vc.wavefile);
    load_stim(is, pperturb, sigadjust, stims);
    for (int i = 0, k = pperturb.size(); i < k; i++) {
        for (int j = 0, l = pperturb[i].size(); j < l; j++) {
            cerr << pperturb[i][j] << " ";
        }
        for (int j = 0, l = pperturb[i].size(); j < l; j++) {
            cerr << sigadjust[i][j] << " ";
        }
		cerr << endl;
		cerr << stims[i] << endl;
	}

	int Nstim = stims.size();
	vector<vector<double> > errbuf;
	vector<double> eb( MAVGBUFSZ );
	vector<double> mavg( Nstim );
	vector<int> epos( Nstim );
	vector<int> initial( Nstim );
	for (int i = 0, k = Nstim; i < k; i++) {
		errbuf.push_back( eb );
		mavg[i] = 0;
		epos[i] = 0;
		initial[i] = 1;
	}

	//-----------------------------------------------------------------
	// build the neuronal circuitery

	NNmodel model;
    modelDefinition( model );
	allocateMem();
    initialize();
    rtdo_init_bridge();
    var_init_fullrange(); // initialize uniformly on large range
    fprintf( stderr, "# neuronal circuitery built, start computation ... \n\n" );

	//------------------------------------------------------------------
	// output general parameters to output file and start the simulation

	fprintf( stderr, "# We are running with fixed time step %f \n", DT );

	int done = 0, sn;
	unsigned int VSize = NPOP*theSize( model.ftype );
	double lt, oldt;
	double pertFac = 0.1;
	int iTN;

    scalar simulatorVars[NVAR], simulatorParams[NPARAM];
    for ( int i = 0; i < NVAR; i++ )
        simulatorVars[i] = variableIni[i];
    for ( int i = 0; i < NPARAM; i++ )
        simulatorParams[i] = aParamIni[i];
    env.setSimulatorVariables(simulatorVars);
    env.setSimulatorParameters(simulatorParams);

    t = 0.0;
	int nextS = 0;
    int generation = 0;
    while (!done && !*stopFlag)
    {
        truevar_init();
        I = stims[nextS];
        iTN = (int)(I.t / DT);
        stepVGHH = I.baseV;
        otHH = t + I.ot;
        oteHH = t + I.ot + I.dur;
        clampGainHH = cfg->vc.gain;
        accessResistanceHH = cfg->vc.resistance;
        lt = 0.0;
        sn = 0;

        env.setWaveform(I);
        env.sync();

        for (int iT = 0; iT < iTN; iT++) {
            oldt = lt;
            IsynGHH = env.nextSample();
            stepTimeGPU( t );
            t += DT;
            lt += DT;
            if ((sn < I.N) && ((oldt < I.st[sn]) && (lt >= I.st[sn]) || (I.st[sn] == 0))) {
                stepVGHH = I.V[sn];
                sn++;
            }
        }

        env.pause();
        logger->wait();

        CHECK_CUDA_ERRORS( cudaMemcpy( errHH, d_errHH, VSize, cudaMemcpyDeviceToHost ) );

        procreatePopPperturb( pertFac, pperturb, sigadjust, errbuf, epos, initial, mavg, nextS, Nstim, logger, generation++ ); //perturb for next step
    }

    logger->halt();

    fprintf( stderr, "DONE\n" );
	return 0;
}
#endif
