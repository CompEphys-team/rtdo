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

#include "rtdo_types.h"

/**
 * @brief Set up the basic RT machinery and launch AI/AO threads.
 */
void rtdo_init(const char *device_file, const char *calibration_file);

/**
 * @brief Set up an AI/AO channel, active from the next rtdo_sync call.
 * @param type
 * @param subdevice_offset
 * @param channel
 * @param range
 * @param reference is one of AREF_GROUND, AREF_COMMON, AREF_DIFF, AREF_OTHER.
 * @param conversion_factor is either nA/V or mV/V and depends on headstage and amplifier gain.
 * @param buffer_size Number of input or output samples in buffer.
 * @return Returns a handle to the channel created, or 0 on failure.
 */
int rtdo_create_channel(enum rtdo_channel_type type,
                         unsigned int subdevice_offset,
                         unsigned int channel,
                         unsigned int range,
                         unsigned int reference,
                         double conversion_factor,
                        int buffer_size);

/**
 * @brief Puts a channel into an active or inactive state asynchronously.
 * @param handle Channel handle returned by rtdo_create_channel
 * @param active Zero to deactivate, positive value to activate, negative value to get value without changing it
 * @return Previous value of the active flag
 */
int rtdo_set_channel_active(int handle, int active);

/**
 * @brief Set the stimulation pattern asynchronously. This method is thread-safe, as data are copied.
 * @param handle
 * @param baseVal
 * @param numsteps
 * @param values
 * @param ms
 */
int rtdo_set_stimulus(int handle, double baseVal, int numsteps, double *values, double *ms, double ms_total);

/**
 * @brief Applies all pending asynchronous changes, empties the input buffer, and (re-)starts input/output.
 */
void rtdo_sync();

float rtdo_channel_get(int handle);
void rtdo_stop(int unused);

#ifdef __cplusplus
}

int inchan_vc_I, outchan_vc_V;

void initexpHH()
{
    // Initialise
    rtdo_init("/dev/comedi0", "/home/felix/projects/rtdo/ni6251.calibrate");
    inchan_vc_I = rtdo_create_channel(DO_CHANNEL_AI, 0, 0, 0, AREF_DIFF, 100, 1024);
    outchan_vc_V = rtdo_create_channel(DO_CHANNEL_AO, 0, 0, 0, AREF_GROUND, 20, 10);
}

void truevar_initexpHH()
{
    // Start a new experiment: Do nothing.
}

void expHH_setstimulus(inputSpec I) {
    rtdo_set_stimulus(outchan_vc_V, I.baseV, I.N, I.V.data(), I.st.data(), I.t);
}

#ifndef _WAVE
void runexpHH(float t)
{
    // Read
    IsynGHH = rtdo_channel_get(inchan_vc_I);
}
#endif // _WAVE

void endexpHH() {
    // Clean up
    rtdo_stop(0);
}

#endif // __cplusplus

#endif // RT_HELPER_H
