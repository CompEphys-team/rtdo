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

void single_var_init_fullrange(int n)
{
    for ( int i = 0; i < NPARAM; i++ ) {
        mparam[i][n] = aParamRange[2*i] + R.n() * (aParamRange[2*i+1] - aParamRange[2*i]); // uniform in allowed interval
    }
    uids[n] = ++latest_uid;
}

void single_var_reinit(int n, double fac) 
{
    for ( int i = 0; i < NPARAM; i++ ) {
        if ( aParamPMult[i] )
            mparam[i][n] *= (1.0 + fac * aParamSigma[i] * RG.n());
        else
            mparam[i][n] += fac * aParamSigma[i] * RG.n();
    }
}

void single_var_reinit_pperturb(int n, double fac, vector<double> &pperturb) 
{
    for ( int i = 0; i < NPARAM; i++ ) {
        if ( R.n() < pperturb[i] ) {
            if ( aParamPMult[i] )
                mparam[i][n] *= (1.0 + fac * aParamSigma[i] * RG.n());
            else
                mparam[i][n] += fac * aParamSigma[i] * RG.n();
        }
    }
}

void copy_var(int src, int trg)
{
    for ( int i = 0; i < NPARAM; i++ ) {
        mparam[i][trg] = mparam[i][src];
    }
    uids[trg] = ++latest_uid;
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
      for ( int i = 0; i < NVAR; i++ ) {
          mvar[i][n] = variableIni[i];
      }
      errHH[n] = 0.0;
  }
  copyStateToDevice();	  
}

void load_stim(istream &is, vector<vector<double>> &pperturb, vector<inputSpec> &stims) {
    double dtmp;
    vector<double> prob;
    inputSpec I;
    char buf[1024];
    while (is.good()) {
        prob.clear();
        while (((is.peek() == '%') || (is.peek() == '\n') || (is.peek() == ' ')) && is.good()) { // remove comments
            is.getline( buf, BUFSZ );
        }
        for (int i = 0; i < NPARAM; i++) {
            is >> dtmp;
            prob.push_back( dtmp );
        }
        is >> I;
        if (is.good()) {
            pperturb.push_back( prob );
            stims.push_back( I );
        }
    }
}
