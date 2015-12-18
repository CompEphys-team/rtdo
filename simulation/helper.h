/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Informatics
              University of Sussex 
              Brighton BN1 9QJ, UK
  
   email to:  t.nowotny@sussex.ac.uk
  
   initial version: 2014-06-26
  
--------------------------------------------------------------------------*/

#include <vector>
#include <string>
#include <istream>
#include <bitset>
#include <limits>

typedef struct {
  double t;
  double ot;
  double dur;
  double baseV;
  int N;
  vector<double> st;
  vector<double> V;
#ifdef _WAVE
  double fit;
#endif
} inputSpec;

double sigGNa;
double sigENa;
double sigmaoff;
double sigmaslope;
double sigmboff;
double sigmbslope;
double sighaoff;
double sighaslope;
double sighboff;
double sighbslope;
double sigGK;
double sigEK;
double signaoff;
double signaslope;
double signboff;
double signbslope;
double sigGl;
double sigEl;
double sigC;

void loadSig(istream &is)
{
  is >> sigGNa;
  is >> sigENa;
  is >> sigmaoff;
  is >> sigmaslope;
  is >> sigmboff;
  is >> sigmbslope;
  is >> sighaoff;
  is >> sighaslope;
  is >> sighboff;
  is >> sighbslope;
  is >> sigGK;
  is >> sigEK;
  is >> signaoff;
  is >> signaslope;
  is >> signboff;
  is >> signbslope;
  is >> sigGl;
  is >> sigEl;
  is >> sigC;
  cerr << sigC;
  assert(is.good());
}
#ifndef _WAVE
ostream &operator<<(ostream &os, inputSpec &I)
{
  os << " " << I.t << " ";
  os << " " << I.ot << " ";
  os << " " << I.dur << " ";
  os << " " << I.baseV << "  ";
  os << " " << I.N << "    ";  
  for (int i= 0; i < I.N; i++) {
    os << I.st[i] << " ";
    os << I.V[i] << "  ";
  }
  return os;
}
#endif
void write_para() 
{
  fprintf(stderr, "# DT %f \n", DT);
}

const double limit[Npara][2]= {{1.0, 500.0}, // gNa
			       {0.0, 100.0}, // ENa
			       {3.25, 3.75}, // maoff +/- 2.5mV/10
			       {0.09, 0.11}, // maslope +/- 10%
			       {57.5, 62.5}, // mboff +/- 2.5mV
			       {16.2, 19.8}, // mbslope +/ 10%
			       {2.75, 3.25}, // haoff +/- 2.5mV/10
			       {18.0, 22.0}, // haslope +/- 10%
			       {2.75, 3.25}, // hboff +/- 2.5mV/10
			       {0.09, 0.11}, // hbslope +/- 10%
			       {1.0, 500.0}, // gKd
			       {-100.0, 0.0}, // EKd
			       {0.25, 0.75}, // naoff +/- 2.5mV/10
			       {0.009, 0.011}, // naslope +/- 10%
			       {57.5, 62.5}, // nboff +/- 2.5mV
			       {72.0, 88.0}, // nbslope +/- 10%
			       {1.0, 500.0}, // gleak
			       {-100.0, 0.0}, // Eleak
			       {1e-1, 10.0}}; // C


void single_var_init_fullrange(int n)
{
  gNaHH[n]= limit[0][0]+R.n()*(limit[0][1]-limit[0][0]); // uniform in allowed interval
  ENaHH[n]= limit[1][0]+R.n()*(limit[1][1]-limit[1][0]); // uniform in allowed interval
  maoffHH[n]= limit[2][0]+R.n()*(limit[2][1]-limit[2][0]); // uniform in allowed interval
  maslopeHH[n]= limit[3][0]+R.n()*(limit[3][1]-limit[3][0]); // uniform in allowed interval
  mboffHH[n]= limit[4][0]+R.n()*(limit[4][1]-limit[4][0]); // uniform in allowed interval
  mbslopeHH[n]= limit[5][0]+R.n()*(limit[5][1]-limit[5][0]); // uniform in allowed interval
  haoffHH[n]= limit[6][0]+R.n()*(limit[6][1]-limit[6][0]); // uniform in allowed interval
  haslopeHH[n]= limit[7][0]+R.n()*(limit[7][1]-limit[7][0]); // uniform in allowed interval
  hboffHH[n]= limit[8][0]+R.n()*(limit[8][1]-limit[8][0]); // uniform in allowed interval
  hbslopeHH[n]= limit[9][0]+R.n()*(limit[9][1]-limit[9][0]); // uniform in allowed interval
  gKHH[n]= limit[10][0]+R.n()*(limit[10][1]-limit[10][0]); // uniform in allowed interval
  EKHH[n]= limit[11][0]+R.n()*(limit[11][1]-limit[11][0]); // uniform in allowed interval
  naoffHH[n]= limit[12][0]+R.n()*(limit[12][1]-limit[12][0]); // uniform in allowed interval
  naslopeHH[n]= limit[13][0]+R.n()*(limit[13][1]-limit[13][0]); // uniform in allowed interval
  nboffHH[n]= limit[14][0]+R.n()*(limit[14][1]-limit[14][0]); // uniform in allowed interval
  nbslopeHH[n]= limit[15][0]+R.n()*(limit[15][1]-limit[15][0]); // uniform in allowed interval
  glHH[n]= limit[16][0]+R.n()*(limit[16][1]-limit[16][0]); // uniform in allowed interval
  ElHH[n]= limit[17][0]+R.n()*(limit[17][1]-limit[17][0]); // uniform in allowed interval
  CHH[n] = limit[18][0] + R.n()*(limit[18][1] - limit[18][0]); // uniform in allowed interval
}

void single_var_reinit(int n, double fac) 
{
  gNaHH[n]*= (1.0+fac*sigGNa*RG.n()); // multiplicative Gaussian noise
  ENaHH[n]+= fac*sigENa*RG.n(); // additive Gaussian noise
  maoffHH[n]+= fac*sigmaoff*RG.n(); // additive Gaussian noise
  maslopeHH[n]*= (1.0+fac*sigmaslope*RG.n()); // multiplicative Gaussian noise
  mboffHH[n]+= fac*sigmboff*RG.n(); // additive Gaussian noise
  mbslopeHH[n]*= (1.0+fac*sigmbslope*RG.n()); // multiplicative Gaussian noise
  haoffHH[n]+= fac*sighaoff*RG.n(); // additive Gaussian noise
  haslopeHH[n]*= (1.0+fac*sighaslope*RG.n()); // multiplicative Gaussian noise
  hboffHH[n]+= fac*sighboff*RG.n(); // additive Gaussian noise
  hbslopeHH[n]*= (1.0+fac*sighbslope*RG.n()); // multiplicative Gaussian noise
  gKHH[n]*= (1.0+fac*sigGK*RG.n()); // multiplicative Gaussian noise
  EKHH[n]+= fac*sigEK*RG.n(); // additive Gaussian noise
  naoffHH[n]+= fac*signaoff*RG.n(); // additive Gaussian noise
  naslopeHH[n]*= (1.0+fac*signaslope*RG.n()); // multiplicative Gaussian noise
  nboffHH[n]+= fac*signboff*RG.n(); // additive Gaussian noise
  nbslopeHH[n]*= (1.0+fac*signbslope*RG.n()); // multiplicative Gaussian noise  
  glHH[n]*= (1.0+fac*sigGl*RG.n()); // multiplicative Gaussian noise
  ElHH[n]+= fac*sigEl*RG.n(); // additive Gaussian noise
  CHH[n]*= (1.0+fac*sigC*RG.n()); // multiplicative Gaussian noise
}

void single_var_reinit_pperturb(int n, double fac, vector<double> &pperturb) 
{
    if (R.n() < pperturb[0]) {
	gNaHH[n]*= (1.0+fac*sigGNa*RG.n()); // multiplicative Gaussian noise
    }
    if (R.n() < pperturb[1]) {
	ENaHH[n]+= fac*sigENa*RG.n(); // additive Gaussian noise
    }
    if (R.n() < pperturb[2]) {
      maoffHH[n]+= fac*sigmaoff*RG.n(); // additive Gaussian noise
    }
    if (R.n() < pperturb[3]) {
      maslopeHH[n]*= (1.0+fac*sigmaslope*RG.n()); // multiplicative Gaussian noise
    }
    if (R.n() < pperturb[4]) {
      mboffHH[n]+= fac*sigmboff*RG.n(); // additive Gaussian noise
    }
    if (R.n() < pperturb[5]) {
      mbslopeHH[n]*= (1.0+fac*sigmbslope*RG.n()); // multiplicative Gaussian noise
    }
    if (R.n() < pperturb[6]) {
      haoffHH[n]+= fac*sighaoff*RG.n(); // additive Gaussian noise
    }
    if (R.n() < pperturb[7]) {
      haslopeHH[n]*= (1.0+fac*sighaslope*RG.n()); // multiplicative Gaussian noise
    }
    if (R.n() < pperturb[8]) {
      hboffHH[n]+= fac*sighboff*RG.n(); // additive Gaussian noise
    }
    if (R.n() < pperturb[9]) {
      hbslopeHH[n]*= (1.0+fac*sighbslope*RG.n()); // multiplicative Gaussian noise
    }
    if (R.n() < pperturb[10]) {
	gKHH[n]*= (1.0+fac*sigGK*RG.n()); // multiplicative Gaussian noise
    }
    if (R.n() < pperturb[11]) {
	EKHH[n]+= fac*sigEK*RG.n(); // additive Gaussian noise
    }
    if (R.n() < pperturb[12]) {
      naoffHH[n]+= fac*signaoff*RG.n(); // additive Gaussian noise
    }
    if (R.n() < pperturb[13]) {
      naslopeHH[n]*= (1.0+fac*signaslope*RG.n()); // multiplicative Gaussian noise
    }
    if (R.n() < pperturb[14]) {
      nboffHH[n]+= fac*signboff*RG.n(); // additive Gaussian noise
    }
    if (R.n() < pperturb[15]) {
      nbslopeHH[n]*= (1.0+fac*signbslope*RG.n()); // multiplicative Gaussian noise  
    }
    if (R.n() < pperturb[16]) {
	glHH[n]*= (1.0+fac*sigGl*RG.n()); // multiplicative Gaussian noise
    }
    if (R.n() < pperturb[17]) {
	ElHH[n]+= fac*sigEl*RG.n(); // additive Gaussian noise
    }
    if (R.n() < pperturb[18]) {
	CHH[n]*= (1.0+fac*sigC*RG.n()); // multiplicative Gaussian noise
    }
}

void copy_var(int src, int trg)
{
  gNaHH[trg]= gNaHH[src];
  ENaHH[trg]= ENaHH[src];
  maoffHH[trg]= maoffHH[src];
  maslopeHH[trg]= maslopeHH[src];
  mboffHH[trg]= mboffHH[src];
  mbslopeHH[trg]= mbslopeHH[src];
  haoffHH[trg]= haoffHH[src];
  haslopeHH[trg]= haslopeHH[src];
  hboffHH[trg]= hboffHH[src];
  hbslopeHH[trg]= hbslopeHH[src];
  gKHH[trg]= gKHH[src];
  EKHH[trg]=EKHH[src];
  naoffHH[trg]=naoffHH[src];
  naslopeHH[trg]=naslopeHH[src];
  nboffHH[trg]=nboffHH[src];
  nbslopeHH[trg]=nbslopeHH[src];
  glHH[trg]= glHH[src];
  ElHH[trg]= ElHH[src];
  CHH[trg]= CHH[src];
}

void var_init_fullrange()
{
  for (int n= 0; n < NPOP; n++) {
    single_var_init_fullrange(n);
  }
  copyStateToDevice();	
}
 
void var_reinit(double fac) 
{
  // add noise to the parameters
  for (int n= 0; n < NPOP; n++) {
    single_var_reinit(n, fac);
  }
  copyStateToDevice();	
}

void truevar_init()
{
  for (int n= 0; n < NPOP; n++) {  
    VHH[n]= myHH_ini[0];
    mHH[n]= myHH_ini[1];
    hHH[n]= myHH_ini[2];
    nHH[n]= myHH_ini[3];
    errHH[n]= 0.0;
  }
  copyStateToDevice();	  
}


double Vexp;
double mexp;
double hexp;
double nexp;
double gNaexp;
double ENaexp;
double maoffexp;
double maslopeexp;
double mboffexp;
double mbslopeexp;
double haoffexp;
double haslopeexp;
double hboffexp;
double hbslopeexp;
double gKexp;
double EKexp;
double naoffexp;
double naslopeexp;
double nboffexp;
double nbslopeexp;
double glexp;
double Elexp;
double Cexp;

#ifdef RTDO
#include "run.h"
#else
void endexpHH(){}

void initexpHH()
{
  Vexp= myHH_ini[0];
  mexp= myHH_ini[1];
  hexp= myHH_ini[2];
  nexp= myHH_ini[3];
  gNaexp= myHH_ini[4];
  ENaexp= myHH_ini[5];
  maoffexp= myHH_ini[6];
  maslopeexp= myHH_ini[7];
  mboffexp= myHH_ini[8];
  mbslopeexp= myHH_ini[9];
  haoffexp= myHH_ini[10];
  haslopeexp= myHH_ini[11];
  hboffexp= myHH_ini[12];
  hbslopeexp= myHH_ini[13];
  gKexp= myHH_ini[14];
  EKexp= myHH_ini[15];
  naoffexp= myHH_ini[16];
  naslopeexp= myHH_ini[17];
  nboffexp= myHH_ini[18];
  nbslopeexp= myHH_ini[19];
  glexp= myHH_ini[20];
  Elexp= myHH_ini[21];
  Cexp= myHH_ini[22]; 
}

void truevar_initexpHH()
{
  Vexp= myHH_ini[0];
  mexp= myHH_ini[1];
  hexp= myHH_ini[2];
  nexp= myHH_ini[3];
}

#ifndef _WAVE
void runexpHH(float t)
{
  // calculate membrane potential
  double Imem;
  unsigned int mt;
  double mdt= DT/50.0;
  for (mt=0; mt < 50; mt++) {
    IsynGHH= 200.0*(stepVGHH-Vexp);
    //    cerr << IsynGHH << " " << Vexp << endl;
    Imem= -(mexp*mexp*mexp*hexp*gNaexp*(Vexp-(ENaexp))+
	    nexp*nexp*nexp*nexp*gKexp*(Vexp-(EKexp))+
	    glexp*(Vexp-(Elexp))-IsynGHH);
    double _a= (maoffexp+maslopeexp*Vexp)/(1.0-exp(-maoffexp-maslopeexp*Vexp));
    double _b= 4.0*exp(-(Vexp+mboffexp)/mbslopeexp);
    mexp+= (_a*(1.0-mexp)-_b*mexp)*mdt;
    _a= 0.07*exp(-Vexp/haslopeexp-haoffexp);
    _b= 1.0 / (exp(-hboffexp-hbslopeexp*Vexp)+1.0);
    hexp+= (_a*(1.0-hexp)-_b*hexp)*mdt;
    _a= (-naoffexp-naslopeexp*Vexp) / (exp(-10.0*naoffexp-10.0*naslopeexp*Vexp)-1.0);
    _b= 0.125*exp(-(Vexp+nboffexp)/nbslopeexp);
    nexp+= (_a*(1.0-nexp)-_b*nexp)*mdt;
    Vexp+= Imem/Cexp*mdt;
  }
  IsynGHH+= INoiseSTD*RG.n();
}
#endif // _WAVE

#endif // RTDO

void initI(inputSpec &I) 
{
  I.t= 200.0;
  I.baseV= -60.0;
  I.N= 12;
  I.st.push_back(10.0);
  I.V.push_back(-30.0);
  I.st.push_back(20.0);
  I.V.push_back(-60.0);

  I.st.push_back(40.0);
  I.V.push_back(-20.0);
  I.st.push_back(50.0);
  I.V.push_back(-60.0);

  I.st.push_back(70.0);
  I.V.push_back(-10.0);
  I.st.push_back(80.0);
  I.V.push_back(-60.0);

  I.st.push_back(100.0);
  I.V.push_back(0.0);
  I.st.push_back(110.0);
  I.V.push_back(-60.0);

  I.st.push_back(130.0);
  I.V.push_back(10.0);
  I.st.push_back(140.0);
  I.V.push_back(-60.0);

  I.st.push_back(160.0);
  I.V.push_back(20.0);
  I.st.push_back(170.0);
  I.V.push_back(-60.0);
  assert((I.N == I.V.size()) && (I.N == I.st.size()));
}

void load_param_values(std::istream &is, const NNmodel &model) {
    const neuronModel& n = nModels[model.neuronType[0]];
    std::string name, type;
    scalar lower, upper;
    std::bitset<NPARAM> bits;
    int j;
    while ( is.good() ) {
        is >> name;
        if ( name.c_str()[0] == '#' ) {
            is.ignore(numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }
        is >> lower >> upper >> type;
        for ( j = 0; j < NPARAM; j++ ) {
            if ( !name.compare(n.varNames[NVAR + j]) ) {
                mrange[j][0] = lower;
                mrange[j][1] = upper;
                mpertmult[j] = bool(type.compare("+"));
                bits.set(j);
                break;
            }
        }
        is.ignore(numeric_limits<std::streamsize>::max(), '\n');
    }
    assert(bits.all());
}
