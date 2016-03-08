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
#include <math.h>

unsigned long long Model::latestUid = 0;

Model::Model(int idx) :
    idx(idx),
    uid(++latestUid),
    parentIdx(parentIdx),
    parentUid(0),
    errDiff(0.0),
    momentum(currentExperiment->stims.size(), 0.0)
{
    for ( int i = 0; i < NPARAM; i++ ) {
        mparam[i][idx] = aParamRange[2*i] + R.n() * (aParamRange[2*i+1] - aParamRange[2*i]);
    }
}

void Model::diff()
{
    if ( parentUid && currentExperiment->models[parentIdx].uid == parentUid ) {
        errDiff = errHH[parentIdx] - errHH[idx];
    }
}

void Model::copy(int parentIdx)
{
    uid = ++latestUid;
    this->parentIdx = parentIdx;
    parentUid = currentExperiment->models[parentIdx].uid;
    errDiff = 0.0;
    momentum = currentExperiment->models[parentIdx].momentum;
    for ( int i = 0; i < NPARAM; i++ ) {
        mparam[i][idx] = mparam[i][parentIdx];
    }
}

void Model::reinit(int stim)
{
    for ( int i = 0; i < NPARAM; i++ ) {
        if ( currentExperiment->pperturb[stim][i] > 0 ) {
            mparam[i][idx] = aParamRange[2*i] + R.n() * (aParamRange[2*i+1] - aParamRange[2*i]);
        }
    }
    momentum[stim] = 0.0;
}

void Model::mutate(int stim, double fac, bool retainMomentum)
{
    if ( !retainMomentum || momentum[stim] == 0.0 ) { // Reset momentum
        momentum[stim] = RG.n() * fac;
    } else { // Adjust momentum
        if ( currentExperiment->models[parentIdx].errDiff < 0 ) {
            // My parent did better than its parent => Keep going in my parent's footsteps
            momentum[stim] *= 2;
        } else if ( currentExperiment->models[parentIdx].errDiff > 0 ) {
            // My parent did worse than its parent => Turn around and go slower
            momentum[stim] *= -0.3;
        }
        // If errDiff == 0 (stim transition), retain momentum without change
    }

    for ( int i = 0; i < NPARAM; i++ ) {
        if ( currentExperiment->pperturb[stim][i] > 0
             && (currentExperiment->pperturb[stim][i] == 1 || R.n() < currentExperiment->pperturb[stim][i]) ) {
            if ( aParamPMult[i] )
                mparam[i][idx] *= (1.0 + aParamSigma[i] * currentExperiment->sigadjust[stim][i] * momentum[stim]);
            else
                mparam[i][idx] += aParamSigma[i] * currentExperiment->sigadjust[stim][i] * momentum[stim];

            if ( mparam[i][idx] < aParamRange[2*i] ) {
                mparam[i][idx] = aParamRange[2*i];
                momentum[stim] = fabs(RG.n()) * fac;
            } else if ( mparam[i][idx] > aParamRange[2*i + 1] ) {
                mparam[i][idx] = aParamRange[2*i + 1];
                momentum[stim] = -fabs(RG.n()) * fac;
            }
        }
    }
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
    char buf[BUFSZ];
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
