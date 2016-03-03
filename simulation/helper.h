/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Informatics
              University of Sussex 
              Brighton BN1 9QJ, UK
  
   email to:  t.nowotny@sussex.ac.uk
  
   initial version: 2014-06-26
  
--------------------------------------------------------------------------*/

#include <vector>
#include <istream>

void single_var_init_fullrange(int n)
{
    for ( int i = 0; i < NPARAM; i++ ) {
        mparam[i][n] = aParamRange[2*i] + R.n() * (aParamRange[2*i+1] - aParamRange[2*i]); // uniform in allowed interval
    }
    uids[n] = ++latest_uid;
}

void single_var_reinit_pperturb_fullrange(int n, vector<double> &pperturb)
{
    for ( int i = 0; i < NPARAM; i++ ) {
        if ( pperturb[i] > 0 && R.n() < pperturb[i] ) {
            mparam[i][n] = aParamRange[2*i] + R.n() * (aParamRange[2*i+1] - aParamRange[2*i]);
        }
    }
}

void single_var_reinit_pperturb(int n, double fac, vector<double> &pperturb, vector<double> &sigadjust)
{
    for ( int i = 0; i < NPARAM; i++ ) {
        if ( pperturb[i] > 0 && R.n() < pperturb[i] ) {
            if ( aParamPMult[i] )
                mparam[i][n] *= (1.0 + fac * aParamSigma[i] * sigadjust[i] * RG.n());
            else
                mparam[i][n] += fac * aParamSigma[i] * sigadjust[i] * RG.n();

            if ( mparam[i][n] < aParamRange[2*i] )
                mparam[i][n] = aParamRange[2*i];
            else if ( mparam[i][n] > aParamRange[2*i + 1] )
                mparam[i][n] = aParamRange[2*i + 1];
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

void load_stim(istream &is, vector<vector<double>> &pperturb, vector<vector<double>> &sigadjust, vector<inputSpec> &stims) {
    double dtmp;
    vector<double> prob;
    vector<double> adjust;
    inputSpec I;
    char buf[1024];
    while (is.good()) {
        prob.clear();
        adjust.clear();
        while (((is.peek() == '%') || (is.peek() == '\n') || (is.peek() == ' ') || (is.peek() == '#')) && is.good()) { // remove comments
            is.getline( buf, BUFSZ );
        }
        for (int i = 0; i < NPARAM; i++) {
            is >> dtmp;
            prob.push_back( dtmp );
        }
        for ( int i = 0; i < NPARAM; i++ ) {
            is >> dtmp;
            adjust.push_back(dtmp);
        }
        is >> I;
        if (is.good()) {
            pperturb.push_back( prob );
            sigadjust.push_back(adjust);
            stims.push_back( I );
        }
    }
}

int compareErrTupel( const void *x, const void *y )
{
    // make sure NaN is largest
    if (std::isnan( ((errTupel *)x)->err )) return 1;
    if (std::isnan( ((errTupel *)y)->err )) return -1;
    if (((errTupel *)x)->err < ((errTupel *)y)->err) return -1;
    if (((errTupel *)x)->err > ( (errTupel *)y )->err) return 1;
    return 0;
}
