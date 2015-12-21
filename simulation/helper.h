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

scalar range[NPARAM][2];
bool pertmult[NPARAM];
double sigma[NPARAM];

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

void single_var_init_fullrange(int n)
{
    for ( int i = 0; i < NPARAM; i++ ) {
        mparam[i][n] = range[i][0] + R.n() * (range[i][1] - range[i][0]); // uniform in allowed interval
    }
}

void single_var_reinit(int n, double fac) 
{
    for ( int i = 0; i < NPARAM; i++ ) {
        if ( pertmult[i] )
            mparam[i][n] *= (1.0 + fac * sigma[i] * RG.n());
        else
            mparam[i][n] += fac * sigma[i] * RG.n();
    }
}

void single_var_reinit_pperturb(int n, double fac, vector<double> &pperturb) 
{
    for ( int i = 0; i < NPARAM; i++ ) {
        if ( R.n() < pperturb[i] ) {
            if ( pertmult[i] )
                mparam[i][n] *= (1.0 + fac * sigma[i] * RG.n());
            else
                mparam[i][n] += fac * sigma[i] * RG.n();
        }
    }
}

void copy_var(int src, int trg)
{
    for ( int i = 0; i < NPARAM; i++ ) {
        mparam[i][trg] = mparam[i][src];
    }
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
          mvar[i][n] = mvar_ini[i];
      }
      errM[n] = 0.0;
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
        I.st.clear();
        I.V.clear();
        while (((is.peek() == '%') || (is.peek() == '\n') || (is.peek() == ' ')) && is.good()) { // remove comments
            is.getline( buf, BUFSZ );
        }
        for (int i = 0; i < NPARAM; i++) {
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
}

void load_param_values(std::istream &is, const NNmodel &model) {
    const neuronModel& n = nModels[model.neuronType[0]];
    std::string name, type;
    scalar lower, upper;
    double sig;
    std::bitset<NPARAM> bits;
    int j;
    while ( is.good() ) {
        is >> name;
        if ( name.c_str()[0] == '#' ) {
            is.ignore(numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }
        is >> lower >> upper >> type >> sig;
        for ( j = 0; j < NPARAM; j++ ) {
            if ( !name.compare(n.varNames[NVAR + j]) ) {
                range[j][0] = lower;
                range[j][1] = upper;
                pertmult[j] = bool(type.compare("+"));
                sigma[j] = sig;
                bits.set(j);
                break;
            }
        }
        is.ignore(numeric_limits<std::streamsize>::max(), '\n').peek();
    }
    assert(bits.all());
}
