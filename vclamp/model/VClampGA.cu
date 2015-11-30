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

int main( int argc, char *argv[] )
{
	if (argc != 6)
	{
		fprintf( stderr, "usage: VClampGA <basename> <stimulus file> <sigma file> <CPU=0, GPU=1> <protocol> \n" );
		return 1;
	}
	int which = atoi( argv[4] );
	int protocol = atoi( argv[5] );
	string stimFName = toString( argv[2] );
	ifstream is( stimFName.c_str() );
	ifstream sis( argv[3] );
	loadSig( sis );

	char buf[BUFSZ];
	int experimentNo = 0;
	string OutDir = toString( argv[1] ) + "_output";
	while (file_exists( OutDir + "/" + toString( argv[1] ) + "_" + std::to_string( experimentNo ) + toString( ".time" ) )) { ++experimentNo; }
	string name;
	name = OutDir + "/" + toString( argv[1] ) + "_" + std::to_string( experimentNo ) + toString( ".time" );
	FILE *timef = fopen( name.c_str(), "a" );

	write_para();

	name = OutDir + "/" + toString( argv[1] ) + "_" + std::to_string( experimentNo ) + toString( ".out.I" );
	FILE *osf = fopen( name.c_str(), "w" );
	name = OutDir + "/" + toString( argv[1] ) + "_" + std::to_string( experimentNo ) + toString( ".out.best" );
	FILE *osb = fopen( name.c_str(), "w" );


	//-----------------------------------------------------------------
	// read the relevant stimulus patterns
	double dtmp;
	vector<double> prob;
	vector<vector<double> > pperturb;
	vector<inputSpec> stims;
	inputSpec I;
	while (is.good()) {
		prob.clear();
		I.st.clear();
		I.V.clear();
		while (((is.peek() == '%') || (is.peek() == '\n') || (is.peek() == ' ')) && is.good()) { // remove comments
			is.getline( buf, BUFSZ );
			cerr << "removed: " << buf << endl;
		}
		for (int i = 0; i < Npara; i++) {
			is >> dtmp;
			prob.push_back( dtmp );
		}
		is >> I.t;
		is >> I.ot;
		is >> I.dur;
		is >> I.baseV;
		is >> I.N;
		for (int i = 0; i < I.N; i++) {
			is >> dtmp;
			I.st.push_back( dtmp );
			is >> dtmp;
			I.V.push_back( dtmp );
		}
		if (is.good()) {
			pperturb.push_back( prob );
			stims.push_back( I );
		}
	}

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

	//-----------------------------------------------------------------
	// build the neuronal circuitery

	NNmodel model;
	modelDefinition( model );
	allocateMem();
	initialize();
	var_init_fullrange(); // initialize uniformly on large range
	initexpHH();
	fprintf( stderr, "# neuronal circuitery built, start computation ... \n\n" );

	double *theExp_p[Npara];
	theExp_p[0] = &gNaexp;
	theExp_p[1] = &ENaexp;
	theExp_p[2] = &gKexp;
	theExp_p[3] = &EKexp;
	theExp_p[4] = &glexp;
	theExp_p[5] = &Elexp;
	theExp_p[6] = &Cexp;

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
	while (!done)
	{
		truevar_init();
		truevar_initexpHH();
		for (int s = 0, k = pperturb.size(); s < k; s++) {
			I = stims[nextS];
			iTN = (int)(I.t / DT);
			stepVGHH = I.baseV;
			otHH = t + I.ot;
			oteHH = t + I.t + I.dur;
			lt = 0.0;
			sn = 0;
#ifdef RTDO
            expHH_setstimulus(I);
            rtdo_sync();
#endif
			for (int iT = 0; iT < iTN; iT++) {
				oldt = lt;
				runexpHH( t );
				if (which == GPU) {
					stepTimeGPU( t );
				}
				else {
					stepTimeCPU( t );
				}
				t += DT;
				lt += DT;
				if ((sn < I.N) && ((oldt < I.st[sn]) && (lt >= I.st[sn]) || (I.st[sn] == 0))) {
					stepVGHH = I.V[sn];
					sn++;
					fprintf( osf, "%f %f \n", t, stepVGHH );
				}
			}
			if (which == 1) {
				CHECK_CUDA_ERRORS( cudaMemcpy( errHH, d_errHH, VSize, cudaMemcpyDeviceToHost ) );
			}
			// output the best values, which input was used and the resulting error
			fprintf( osb, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %d ", t, gNaexp, ENaexp, maoffexp, maslopeexp, mboffexp, mbslopeexp, haoffexp, haslopeexp, hboffexp, hbslopeexp, gKexp, EKexp, naoffexp, naslopeexp, nboffexp, nbslopeexp, glexp, Elexp, Cexp, nextS );
			procreatePopPperturb( osb, pertFac, pperturb, errbuf, epos, initial, mavg, nextS, Nstim ); //perturb for next step
		}
		if (protocol >= 0) {
			if (protocol < Npara) {
				if (protocol % 2 == 0) {
					*(theExp_p[protocol]) = myHH_ini[protocol + 4] * (1 + 0.5*sin( 3.1415927*t / 40000 ));
				}
				else {
					*(theExp_p[protocol]) = myHH_ini[protocol + 4] + 40.0*(sin( 3.1415927*t / 40000 ));
				}
			}
			else {
				for (int pn = 0; pn < Npara; pn++) {
					double fac;
					if (pn % 2 == 0) {
						fac = 1 + 0.005*RG.n();
						*(theExp_p[pn]) *= fac;
					}
					else {
						fac = 0.04*RG.n();
						*(theExp_p[pn]) += fac;
					}
				}
			}
		}
		cerr << "% " << t << endl;
		done = (t >= TOTALT);
	}
    endexpHH();
	timer.stopTimer();
	fprintf( timef, "%f \n", timer.getElapsedTime() );
	// close files 
	fclose( osf );
	fclose( timef );
	fclose( osb );

	fprintf( stderr, "DONE" );
	return 0;
}
#endif
