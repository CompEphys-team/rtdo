/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2015-12-08

--------------------------------------------------------------------------*/

#ifndef RUN_H
#define RUN_H

#include "types.h"
#include "rt.h"
#include <iostream>

void run_vclamp_start();
void run_vclamp_stop();

daq_channel *run_get_active_outchan();
daq_channel *run_get_active_inchan();

void run_digest(double best_err, double mavg, int nextS);
int run_check_break();

#ifdef RTDO // Included from within model/helper.h
void endexpHH() {}
void initexpHH() {}
void truevar_initexpHH() {}
void runexpHH(float t) {
    daq_channel *in = run_get_active_inchan();
    int err=0;
    IsynGHH = (double)rtdo_get_data(in->handle, &err);
}
void expHH_setstimulus(inputSpec I) {
    daq_channel *out = run_get_active_outchan();
    int ret = rtdo_set_stimulus(out->handle, I.baseV, I.N, I.V.data(), I.st.data(), I.t);
    if ( ret )
        std::cerr << "Error " << ret << " setting stimulus waveform" << std::endl;
}
#endif

#endif // RUN_H
