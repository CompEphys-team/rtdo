/*--------------------------------------------------------------------------
Author: Thomas Nowotny

Institute: Informatics
University of Sussex
Brighton BN1 9QJ, UK

email to:  t.nowotny@sussex.ac.uk

initial version: 2014-06-25

--------------------------------------------------------------------------*/

#include <algorithm>
#include <cmath>

typedef struct
{
	unsigned int id;
	double err;
} errTupel;

int compareErrTupel( const void *x, const void *y )
{
	// make sure NaN is largest
	if (std::isnan( ((errTupel *)x)->err )) return 1;
	if (std::isnan( ((errTupel *)y)->err )) return -1;
	if (((errTupel *)x)->err < ((errTupel *)y)->err) return -1;
	if (((errTupel *)x)->err > ( (errTupel *)y )->err) return 1;
	return 0;
}


void procreatePopPperturb( FILE *osb, double amplitude, vector<vector<double> > &pperturb, vector<vector<double> > &errbuf, vector<int> &epos, vector<int> &initial, vector<double> &mavg, int &nextS, int Nstim )
{
	double tmavg;
	static errTupel errs[NPOP];
	for (int i = 0; i < NPOP; i++) {
		errs[i].id = i;
        errs[i].err = errM[i];
	}
	qsort( (void *)errs, NPOP, sizeof( errTupel ), compareErrTupel );
//#define DEBUG_PROCREATE
#ifdef DEBUG_PROCREATE
	cerr << "% sorted fitness: ";
	for (int i = 0; i < NPOP; i++) {
		cerr << errs[i].err << " ";
	}
	cerr << endl;
#endif
    //fprintf( osb, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f \n", gNaHH[errs[0].id], ENaHH[errs[0].id], maoffHH[errs[0].id], maslopeHH[errs[0].id], mboffHH[errs[0].id], mbslopeHH[errs[0].id], haoffHH[errs[0].id], haslopeHH[errs[0].id], hboffHH[errs[0].id], hbslopeHH[errs[0].id], gKHH[errs[0].id], EKHH[errs[0].id], naoffHH[errs[0].id], naslopeHH[errs[0].id], nboffHH[errs[0].id], nbslopeHH[errs[0].id], glHH[errs[0].id], ElHH[errs[0].id], CHH[errs[0].id], errHH[errs[0].id] );

	// update moving averages
	epos[nextS] = (epos[nextS] + 1) % MAVGBUFSZ;
	if (initial[nextS]) {
		if (epos[nextS] == 0) {
			initial[nextS] = 0;
		}
	}
	else {
		mavg[nextS] -= errbuf[nextS][epos[nextS]];
	}
	errbuf[nextS][epos[nextS]] = errs[0].err;
	mavg[nextS] += errbuf[nextS][epos[nextS]];
	tmavg = mavg[nextS] / MAVGBUFSZ;
	if (errs[0].err < tmavg*0.8) {
		// we are getting better on this one -> adjust a different parameter combination
		nextS = (nextS + 1) % Nstim;
	}

    run_digest(errs[0].err, tmavg, nextS);

	// mutate the second half of the instances
	int k = NPOP / 3;
	for (int i = k; i < 2 * k; i++) {
		copy_var( errs[i - k].id, errs[i].id ); // copy good ones over bad ones
		single_var_reinit_pperturb( errs[i].id, amplitude, pperturb[nextS] ); // jiggle the new copies a bit
	}
	for (int i = 2 * k; i < NPOP; i++)
	{
		single_var_init_fullrange( errs[i].id );
	}
	copyStateToDevice();
}
