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
#include "shared.h"

int compareErrTupel( const void *x, const void *y )
{
	// make sure NaN is largest
	if (std::isnan( ((errTupel *)x)->err )) return 1;
	if (std::isnan( ((errTupel *)y)->err )) return -1;
	if (((errTupel *)x)->err < ((errTupel *)y)->err) return -1;
	if (((errTupel *)x)->err > ( (errTupel *)y )->err) return 1;
	return 0;
}


void procreatePopPperturb( double amplitude,
                           vector<vector<double> > &pperturb,
                           vector<vector<double> > &errbuf,
                           vector<int> &epos,
                           vector<int> &initial,
                           vector<double> &mavg,
                           int &nextS,
                           int Nstim,
                           backlog::BacklogVirtual *logger,
                           int generation )
{
    double tmavg, delErr;
	static errTupel errs[NPOP];
    static int limiter = 0;
	for (int i = 0; i < NPOP; i++) {
		errs[i].id = i;
        errs[i].err = errHH[i];
	}
	qsort( (void *)errs, NPOP, sizeof( errTupel ), compareErrTupel );

    int k = NPOP / 3;
    logger->touch(&errs[0], &errs[k-1], generation, nextS);

	// update moving averages
	epos[nextS] = (epos[nextS] + 1) % MAVGBUFSZ;
	if (initial[nextS]) {
		if (epos[nextS] == 0) {
			initial[nextS] = 0;
		}
        delErr = 2 * errs[0].err;
	}
	else {
        delErr = errbuf[nextS][epos[nextS]];
        mavg[nextS] -= delErr;
	}
	errbuf[nextS][epos[nextS]] = errs[0].err;
	mavg[nextS] += errbuf[nextS][epos[nextS]];
	tmavg = mavg[nextS] / MAVGBUFSZ;

    if (errs[0].err < tmavg*0.8) {
		// we are getting better on this one -> adjust a different parameter combination
		nextS = (nextS + 1) % Nstim;
    }
    if ( delErr > 1.01 * errs[0].err ) {
        limiter = (limiter < 3 ? 0 : limiter-3);
    }
    if ( ++limiter > 5 ) {
        // Stuck, move on
        nextS = (nextS + 1) % Nstim;
        limiter = 0;
    }

    run_digest(generation, errs[0].err, tmavg, nextS);

    // mutate the second half of the instances
	for (int i = k; i < 2 * k; i++) {
		copy_var( errs[i - k].id, errs[i].id ); // copy good ones over bad ones
		single_var_reinit_pperturb( errs[i].id, amplitude, pperturb[nextS] ); // jiggle the new copies a bit
	}
	for (int i = 2 * k; i < NPOP; i++)
	{
		single_var_init_fullrange( errs[i].id );
    }
}
