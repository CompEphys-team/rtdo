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

extern "C" int vclamp(const char *basename, const char *outdir, const char *stimFName) {
    ifstream is( stimFName );

	int experimentNo = 0;
    string OutDir = toString(outdir);
    while (file_exists( OutDir + "/" + toString( basename ) + "_" + std::to_string( experimentNo ) + toString( ".time" ) )) { ++experimentNo; }
	string name;
    name = OutDir + "/" + toString( basename ) + "_" + std::to_string( experimentNo ) + toString( ".time" );
    FILE *timef = fopen( name.c_str(), "a" );
    name = OutDir + "/" + toString( basename ) + "_" + std::to_string( experimentNo ) + toString( ".out.I" );
	FILE *osf = fopen( name.c_str(), "w" );
    name = OutDir + "/" + toString( basename ) + "_" + std::to_string( experimentNo ) + toString( ".out.best" );
	FILE *osb = fopen( name.c_str(), "w" );


	//-----------------------------------------------------------------
    // read the relevant stimulus patterns
	vector<vector<double> > pperturb;
	vector<inputSpec> stims;
    inputSpec I;
    load_stim(is, pperturb, stims);
	for (int i = 0, k = pperturb.size(); i < k; i++) {
		for (int j = 0, l = pperturb[i].size(); j < l; j++) {
			cerr << pperturb[i][j] << " ";
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

    backlog::AsyncLog *logger = new backlog::AsyncLog(Nstim*NPOP, Nstim);
    run_use_backlog(logger->log());

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

	timer.startTimer();
	t = 0.0; cerr << pperturb.size() << " iterations" << endl;
	int nextS = 0;
    int generation = 0;
	while (!done)
    {
        truevar_init();
        for (int s = 0, k = pperturb.size(); s < k; s++) {
			I = stims[nextS];
			iTN = (int)(I.t / DT);
            stepVGHH = I.baseV;
            otHH = t + I.ot;
            oteHH = t + I.ot + I.dur;
			lt = 0.0;
            sn = 0;

            run_setstimulus(I);
            rtdo_sync();

            for (int iT = 0; iT < iTN; iT++) {
				oldt = lt;
                IsynGHH = run_getsample( t );
                stepTimeGPU( t );
				t += DT;
				lt += DT;
				if ((sn < I.N) && ((oldt < I.st[sn]) && (lt >= I.st[sn]) || (I.st[sn] == 0))) {
                    stepVGHH = I.V[sn];
					sn++;
                    fprintf( osf, "%f %f \n", t, stepVGHH );
                }
            }
            CHECK_CUDA_ERRORS( cudaMemcpy( errHH, d_errHH, VSize, cudaMemcpyDeviceToHost ) );

            if ( (done = run_check_break()) ) {
                break;
            }

            procreatePopPperturb( osb, pertFac, pperturb, errbuf, epos, initial, mavg, nextS, Nstim, logger, generation++ ); //perturb for next step
        }
	}
    timer.stopTimer();
	fprintf( timef, "%f \n", timer.getElapsedTime() );
	// close files 
	fclose( osf );
	fclose( timef );
	fclose( osb );

    logger->halt();

	fprintf( stderr, "DONE" );
	return 0;
}
#endif
