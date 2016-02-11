#include <algorithm>
#include "run.h"

#define GA_CROSSOVER_PROB 0.85
#define GA_MUTATE_PROB 0.5


void single_var_init_exact( int n )
{
    for ( int i = 0; i < NPARAM; i++ ) {
        mparam[i][n] = aParamIni[i];
    }
}

void single_var_init_auto_detune( int n )
{
    int p = n % (NPARAM+1) - 1;
    for ( int i = 0; i < NPARAM; i++ ) {
        mparam[i][n] = aParamIni[i] * (1.0 + 0.01 * (p == i));
    }
}

void var_init_exact()
{
	for (int n = 0; n < NPOP; ++n) {
		single_var_init_exact( n );
	}
	copyStateToDevice();
}

void var_init_auto_detune()
{
	for (int n = 0; n < NPOP; ++n) {
		single_var_init_auto_detune( n );
	}
	copyStateToDevice();
}

void wave_pop_init( std::vector<inputSpec> & stims, int N )
{
	inputSpec I;
    double st, nst, theV;
    I.t = TOTALT;
    I.ot = OT;
    I.baseV = VSTEP0;
    I.N = NVSTEPS;
	for (int i = 0; i < N; ++i) {
	tryagain:
		I.st.clear();
		I.V.clear();
		st = 0.0;
		for (int j = 0; j < I.N; j++) {
			do {
				nst = R.n()*STEPWDINI;
				if (TOTALT - st < MINSTEP) { goto tryagain; }
			} while ((nst < MINSTEP) || (st + nst > TOTALT));
			st += nst;
			I.st.push_back( st );
			theV = RG.n()*VSTEPINI + VSTEP0; // steps around resting with Gaussian distribution
            if (theV < MINV) theV = MINV;
			if (theV > MAXV) theV = MAXV;
			I.V.push_back( theV );
        }
        I.dur = R.n() * (MAXT-MINT) + MINT;
		stims.push_back( I );
	}
}

bool larger( inputSpec i, inputSpec j )
{
	if (std::isfinite( i.fit ) && !std::isfinite( j.fit ))
	{
		return true;
	}
	return i.fit > j.fit;
}

inputSpec mutate( inputSpec &inI )
{
	inputSpec I;
	double nst, theV, theOBST;

	I = inI;
	for (int i = 0; i < inI.N; i++) {
		do {
			nst = I.st[i] + mutateA*RG.n();
		} while ((nst < MINSTEP)
			|| ((i > 0) && (nst - I.st[i - 1] < MINSTEP))
			|| ((i < inI.N - 1) && (I.st[i + 1] - nst < MINSTEP))
			|| (nst > TOTALT));
		I.st[i] = nst;
		do {
			theV = I.V[i] + mutateA*RG.n();
		} while ((theV < MINV) || (theV > MAXV));
		I.V[i] = theV;
    }
	do {
        theOBST = I.dur + mutateA*RG.n();
	} while ((theOBST < MINT) || (theOBST > MAXT));
    I.dur = theOBST;
	return I;
}

void procreatePop( vector<inputSpec> &pGen )
{
    vector<inputSpec> newGen;
    int k = pGen.size() / 3;

    sort(pGen.begin(), pGen.end(), larger);

    for (int i = 0; i < k; i++) {
        newGen.push_back(pGen[i]);
    }
    for (int i = k; i < 2 * k; i++) {
        newGen.push_back(mutate(pGen[i - k]));
    }
    wave_pop_init(newGen, pGen.size() - 2 * k);
    pGen = newGen;
    assert(pGen.size() == GAPOP);
}
