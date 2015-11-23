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

void rtdo_init();
void rtdo_sync();
float rtdo_get_Im(float t);
void rtdo_set_Vout(float V);
void rtdo_stop(int unused);

#ifdef __cplusplus
}

void initexpHH()
{
    // Initialise
    rtdo_init();
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
    rtdo_set_Vout(stepVGHH);
    IsynGHH = rtdo_get_Im(t);
}
#endif // _WAVE

void endexpHH() {
    // Clean up
    rtdo_stop(0);
}

#endif // __cplusplus

#endif // RT_HELPER_H
