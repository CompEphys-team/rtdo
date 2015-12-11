/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2015-12-03

--------------------------------------------------------------------------*/
#ifndef GLOBALS_H
#define GLOBALS_H

#include "types.h"

#ifdef __cplusplus
#include <string>

extern struct _sim_params {
    std::string vc_wavefile;
    std::string sigfile;
    std::string outdir;
    std::string modeldir;
    int nPop;
    double dt;
} sim_params;
#define SIMPARAM_DEFAULT {"", "", "", "", 1000, 0.25}

extern "C" {
#endif

extern daq_channel daqchan_vout;
extern daq_channel daqchan_cout;
extern daq_channel daqchan_vin;
extern daq_channel daqchan_cin;

#ifdef __cplusplus
}
#endif

#endif // GLOBALS_H
