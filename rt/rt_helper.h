/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2015-11-17

--------------------------------------------------------------------------*/

#ifndef RT_HELPER_H
#define RT_HELPER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "options.h"

void rtdo_init();
void rtdo_create_channel(rtdo_channel_options *);
void rtdo_sync();
float rtdo_channel_get(rtdo_channel_options *chan);
void rtdo_channel_set(rtdo_channel_options *chan, float V);
void rtdo_stop(int unused);

#ifdef __cplusplus
}

void initexpHH()
{
    // Initialise
    rtdo_init();
    rtdo_create_channel(&inchan_vclamp_Im);
    rtdo_create_channel(&outchan_vclamp_Vc);
}

void truevar_initexpHH()
{
    // Start a new experiment
    rtdo_sync();
}

#ifndef _WAVE
void runexpHH(float t)
{
    // Read/Write
    rtdo_channel_set(&outchan_vclamp_Vc, stepVGHH);
    IsynGHH = rtdo_channel_get(&inchan_vclamp_Im);
}
#endif // _WAVE

void endexpHH() {
    // Clean up
    rtdo_stop(0);
}

#endif // __cplusplus

#endif // RT_HELPER_H
