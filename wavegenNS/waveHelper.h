#include <algorithm>
#include "run.h"

#define GA_CROSSOVER_PROB 0.85
#define GA_MUTATE_PROB 0.5

#define NNOVELTY 6
struct noveltyBundle {
public:
    inputSpec wave;
    double novelty[NNOVELTY];

    noveltyBundle() :
        wave {},
        novelty {0}
    {}
    noveltyBundle(int index) :
        novelty { exceedHH[index],
                  (double) nExceedHH[index],
                  bestHH[index],
                  (double) nBestHH[index],
                  separationHH[index],
                  (double) nSeparationHH[index]
                }
    {}

    inline double fitness() const
    {
        return (novelty[4] * DT) * (novelty[5] * DT / (TOTALT - OT));
    }
};

void single_var_init_exact( int n )
{
    for ( int i = 0; i < NPARAM; i++ ) {
        mparam[i][n] = aParamIni[i];
    }
}

void single_var_init_auto_detune( int n, double fac )
{
    int p = n % (NPARAM+1) - 1;
    for ( int i = 0; i < NPARAM; i++ ) {
        mparam[i][n] = aParamIni[i];
        if ( p == i ) {
            if ( aParamPMult[i] )
                mparam[i][n] *= (1.0 + fac * aParamSigma[i] * sigmaAdjust[i]);
            else
                mparam[i][n] += fac * aParamSigma[i] * sigmaAdjust[i];
        }
    }
}

void var_init_exact()
{
	for (int n = 0; n < NPOP; ++n) {
		single_var_init_exact( n );
	}
	copyStateToDevice();
}

void var_init_auto_detune(double fac = 1.0)
{
	for (int n = 0; n < NPOP; ++n) {
        single_var_init_auto_detune( n, fac );
	}
	copyStateToDevice();
}

void reset(scalar *holding, double fac)
{
    for ( int j = 0; j < NPOP; j++ ) {
        for ( int i = 0; i < NVAR; i++ )
            mvar[i][j] = holding[i];
        errHH[j] = 0.0;

        exceedHH[j] = 0;
        exceedCurrentHH[j] = 0;
        nExceedHH[j] = 0;
        nExceedCurrentHH[j] = 0;

        bestHH[j] = 0;
        bestCurrentHH[j] = 0;
        nBestHH[j] = 0;
        nBestCurrentHH[j] = 0;

        separationHH[j] = 0;
        sepCurrentHH[j] = 0;
        nSeparationHH[j] = 0;
        nSepCurrentHH[j] = 0;

        tStartHH[j] = 0;
        tStartCurrentHH[j] = 0;
        tEndHH[j] = 0;

        stepVGHH[j] = VSTEP0;
        single_var_init_auto_detune( j, fac );
    }
    copyStateToDevice();
}

double noveltyDistance(const noveltyBundle &a, const noveltyBundle &b)
{
    double dist;
    for ( int i = 0; i < NNOVELTY; i++ ) {
        double d = a.novelty[i] - b.novelty[i];
        dist += d*d;
    }
    return sqrt(dist);
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
        I.dur = 0;
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

bool fittestNovelty(const noveltyBundle &a, const noveltyBundle &b)
{
    return a.fitness() > b.fitness();
}

inputSpec mutate( const inputSpec &inI )
{
	inputSpec I;
    double nst, theV;

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
	return I;
}

void crossover( inputSpec const& parent1, inputSpec const& parent2, inputSpec & child1, inputSpec & child2 )
{
	child1.t = TOTALT;
	child2.t = TOTALT;
    child1.ot = OT;
    child2.ot = OT;
    child1.dur = 0;
    child2.dur = 0;
    child1.baseV = VSTEP0;
    child2.baseV = VSTEP0;
	child1.N = NVSTEPS;
	child2.N = NVSTEPS;

	std::vector<bool> mutations;
	mutations.reserve( parent1.V.size() + 1 );
	bool offset = R.n() > 0.5;
	for (size_t i = offset; i < parent1.V.size() + 1 + offset; ++i)
	{
		mutations.push_back( i % 2 );
	}

	//We can always switch around voltages
	for (size_t i = 0; i < parent1.V.size(); ++i)
	{
		int rndN = R.n() * (0.1 + mutations.size() - 1);
		if (mutations[rndN])
		{
			child1.V.push_back( parent1.V[i] );
			child2.V.push_back( parent2.V[i] );
		}
		else
		{
			child1.V.push_back( parent2.V[i] );
			child2.V.push_back( parent1.V[i] );
		}
		mutations.erase( mutations.begin() + rndN );
	}
    int rndN = R.n() * (0.1 + mutations.size() - 1);
	mutations.erase( mutations.begin() + rndN );


	//For times we will need to reorder them and readjust
	for (size_t i = 0; i < parent1.V.size(); ++i)
	{
		if (R.n() > 0.5)
		{
			child1.st.push_back( parent1.st[i] );
			child2.st.push_back( parent2.st[i] );
		}
		else
		{
			child1.st.push_back( parent2.st[i] );
			child2.st.push_back( parent1.st[i] );
		}
	}
	std::sort( child1.st.begin(), child1.st.end() );
	std::sort( child2.st.begin(), child2.st.end() );

	for (size_t i = 1; i < child1.st.size(); ++i)
	{
		if (child1.st[i] - child1.st[i - 1] < MINSTEP) // Don't worry too much and just copy whole parent
		{
			child1.st = parent1.st;
			break;
		}
	}
	for (size_t i = 1; i < child2.st.size(); ++i)
	{
		if (child2.st[i] - child2.st[i - 1] < MINSTEP) // Don't worry too much and just copy whole parent
		{
			child2.st = parent2.st;
			break;
		}
	}
}

void procreatePop( vector<inputSpec> &pGen )
{
	vector<inputSpec> newGen;
	int k = pGen.size() / 10;

	sort( pGen.begin(), pGen.end(), larger );

    //10% Elites, and 10% mutated elites to (hopefully) not extinguish recent noveltyDB admissions
	for (int i = 0; i < k; i++) {
		newGen.push_back( pGen[i] );
        newGen.push_back(mutate(pGen[i]));
	}
	//10% New seeds
	wave_pop_init( newGen, k );

	//roulette
	int capacity = (GAPOP + 1) * GAPOP / 2;

	while (newGen.size() < GAPOP) {
		int rand1 = R.n() * capacity;
		int rand2 = R.n() * capacity;
		if (rand1 > rand2) { double tmp = rand1; rand1 = rand2; rand2 = tmp; }
		int parent1 = -1;
		int parent2 = -1;
		for (size_t i = 0; i < GAPOP; i++) //This could probably be approximated/sped up but w/e
		{
			rand1 -= GAPOP - i;
			rand2 -= GAPOP - i;
			if (rand1 <= 0 && parent1 == -1)
			{
				parent1 = i;
			}
			if (rand2 <= 0)
			{
				parent2 = i;
				break;
			}
		}
		inputSpec child1;
		inputSpec child2;
		//Crossover
		if (R.n() < GA_CROSSOVER_PROB)
		{
            if (parent1 == parent2)
			{
				parent2 += (parent2 == GAPOP - 1) ? -1 : 1;
			}
			crossover( pGen[parent1], pGen[parent2], child1, child2 );
		}
		else
		{
			child1 = pGen[parent1];
			child2 = pGen[parent2];
			if (parent1 < k)
			{
				child1 = mutate( pGen[parent1] );
			}
			if (parent2 < k)
			{
				child2 = mutate( pGen[parent2] );
			}
		}
		//Mutate
		if (R.n() < GA_MUTATE_PROB)
		{
			child1 = mutate( child1 );
		}
		if (R.n() < GA_MUTATE_PROB)
		{
			child2 = mutate( child2 );
		}
		newGen.push_back( child1 );
		if (newGen.size() < GAPOP)
		{
			newGen.push_back( child2 );
		}

	}
	pGen = newGen;
	assert( pGen.size() == GAPOP );
}

void wave_pop_init_from(std::vector<inputSpec> &stims, int N, const std::vector<inputSpec> &initial)
{
    if ( initial.size() ) {
        if ( N > initial.size() ) {
            // Add one unchanged copy
            for ( const inputSpec &s : initial ) {
                stims.push_back(s);
            }

            // Pad with mutated copies
            double p = initial.size() * 1.0 / (N - initial.size());
            if ( p > 1 )
                p = 1; // Disregard lower-ranked initials if there isn't enough space
            for ( int i = 0; i < N - initial.size(); i++ ) {
                stims.push_back(mutate(initial.at(i * p)));
            }
        } else {
            // Add one unchanged copy for the top N initials
            for ( int i = 0; i < N; i++ ) {
                stims.push_back(initial.at(i));
            }
        }
    } else {
        wave_pop_init(stims, N);
    }
}

void procreateInitialisedPop(std::vector<inputSpec> &pGen, const std::vector<inputSpec> &initial)
{
    vector<inputSpec> newGen;
    newGen.reserve(pGen.size());
    int k = pGen.size() / 10;

    sort( pGen.begin(), pGen.end(), larger );

    //10% Elites
    for (int i = 0; i < k; i++) {
        newGen.push_back( pGen[i] );
    }

    //10% New seeds from DB, mutated (possibly twice mutated)
    vector<inputSpec> fromInitial;
    fromInitial.reserve(k);
    wave_pop_init_from( fromInitial, k, initial );
    for ( int i = 0; i < k; i++ ) {
        newGen.push_back( mutate(fromInitial[i]) );
    }

    //50% Mutate with quadratic preference for fit waves
    for ( int i = 0; i < 5*k; i++ ) {
        double r = R.n();
        newGen.push_back( mutate(pGen.at(r*r * pGen.size())) );
    }

    //30% crossovers with quadratic preference for fit waves
    while ( newGen.size() < GAPOP ) {
        double p1 = R.n(), p2 = R.n();
        p1 *= p1 * pGen.size();
        p2 *= p2 * pGen.size();
        if ( p1 > p2 )
            swap(p1, p2);
        if ( p1 == p2 )
            p2 += (p2 == GAPOP - 1) ? -1 : 1;
        inputSpec child1, child2;
        crossover( pGen[p1], pGen[p2], child1, child2 );
        newGen.push_back(child1);
        if ( newGen.size() < GAPOP )
            newGen.push_back(child2);
    }

    pGen = newGen;
    assert( pGen.size() == GAPOP );
}
