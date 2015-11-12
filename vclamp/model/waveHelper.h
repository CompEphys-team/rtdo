
#define GA_CROSSOVER_PROB 0.85
#define GA_MUTATE_PROB 0.5


void single_var_init_exact( int n )
{
	gNaHH[n] = myHH_ini[4 + 0];
	ENaHH[n] = myHH_ini[4 + 1];
	maoffHH[n] = myHH_ini[4 + 2];
	maslopeHH[n] = myHH_ini[4 + 3];
	mboffHH[n] = myHH_ini[4 + 4];
	mbslopeHH[n] = myHH_ini[4 + 5];
	haoffHH[n] = myHH_ini[4 + 6];
	haslopeHH[n] = myHH_ini[4 + 7];
	hboffHH[n] = myHH_ini[4 + 8];
	hbslopeHH[n] = myHH_ini[4 + 9];
	gKHH[n] = myHH_ini[4 + 10];
	EKHH[n] = myHH_ini[4 + 11];
	naoffHH[n] = myHH_ini[4 + 12];
	naslopeHH[n] = myHH_ini[4 + 13];
	nboffHH[n] = myHH_ini[4 + 14];
	nbslopeHH[n] = myHH_ini[4 + 15];
	glHH[n] = myHH_ini[4 + 16];
	ElHH[n] = myHH_ini[4 + 17];
	CHH[n] = myHH_ini[4 + 18];
}

void single_var_init_auto_detune( int n )
{
	gNaHH[n] = myHH_ini[4 + 0] * (1.00 + 0.01 * (n % (Npara + 1) == 1));
	ENaHH[n] = myHH_ini[4 + 1] * (1.00 + 0.01 * (n % (Npara + 1) == 2));
	maoffHH[n] = myHH_ini[4 + 2] * (1.00 + 0.01 * (n % (Npara + 1) == 3));
	maslopeHH[n] = myHH_ini[4 + 3] * (1.00 + 0.01 * (n % (Npara + 1) == 4));
	mboffHH[n] = myHH_ini[4 + 4] * (1.00 + 0.01 * (n % (Npara + 1) == 5));
	mbslopeHH[n] = myHH_ini[4 + 5] * (1.00 + 0.01 * (n % (Npara + 1) == 6));
	haoffHH[n] = myHH_ini[4 + 6] * (1.00 + 0.01 * (n % (Npara + 1) == 7));
	haslopeHH[n] = myHH_ini[4 + 7] * (1.00 + 0.01 * (n % (Npara + 1) == 8));
	hboffHH[n] = myHH_ini[4 + 8] * (1.00 + 0.01 * (n % (Npara + 1) == 9));
	hbslopeHH[n] = myHH_ini[4 + 9] * (1.00 + 0.01 * (n % (Npara + 1) == 10));
	gKHH[n] = myHH_ini[4 + 10] * (1.00 + 0.01 * (n % (Npara + 1) == 11));
	EKHH[n] = myHH_ini[4 + 11] * (1.00 + 0.01 * (n % (Npara + 1) == 12));
	naoffHH[n] = myHH_ini[4 + 12] * (1.00 + 0.01 * (n % (Npara + 1) == 13));
	naslopeHH[n] = myHH_ini[4 + 13] * (1.00 + 0.01 * (n % (Npara + 1) == 14));
	nboffHH[n] = myHH_ini[4 + 14] * (1.00 + 0.01 * (n % (Npara + 1) == 15));
	nbslopeHH[n] = myHH_ini[4 + 15] * (1.00 + 0.01 * (n % (Npara + 1) == 16));
	glHH[n] = myHH_ini[4 + 16] * (1.00 + 0.01 * (n % (Npara + 1) == 17));
	ElHH[n] = myHH_ini[4 + 17] * (1.00 + 0.01 * (n % (Npara + 1) == 18));
	CHH[n] = myHH_ini[4 + 18] * (1.00 + 0.01 * (n % (Npara + 1) == 19));
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
	double st, nst, theV, theOBST;
	I.t = TOTALT;
	I.N = NVSTEPS;  // two steps for now
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
		do {
			theOBST = R.n() * 100;
		} while ((theOBST < MINT) || (theOBST > MAXT));
		I.ot = theOBST;
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
		theOBST = I.ot + mutateA*RG.n();
	} while ((theOBST < MINT) || (theOBST > MAXT));
	I.ot = theOBST;
	return I;
}

void crossover( inputSpec const& parent1, inputSpec const& parent2, inputSpec & child1, inputSpec & child2 )
{
	child1.t = TOTALT;
	child2.t = TOTALT;
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
	//And observation time
	if (R.n() > 0.5)
	{
		child1.ot = parent1.ot;
		child2.ot = parent2.ot;
	}
	else
	{
		child2.ot = parent1.ot;
		child1.ot = parent2.ot;
	}
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
#ifdef DEBUG_PROCREATE
	cerr << "% sorted fitness: ";
	for (int i = 0; i < pGen.size(); i++) {
		cerr << pGen[i].fit << " ";
	}
	cerr << endl;
#endif
	//10% Elites
	for (int i = 0; i < k; i++) {
		newGen.push_back( pGen[i] );
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
			if (parent1 = parent2)
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

ostream &operator<<(ostream &os, inputSpec &I)
{
	for (int i = 0; i < I.N; i++) {
		os << I.st[i] << " ";
		os << I.V[i] << "  ";
	}
	os << " " << I.ot << "  ";
	os << " " << I.fit << "    ";
	return os;
}