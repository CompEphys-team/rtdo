/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2015-11-17

--------------------------------------------------------------------------*/

#ifndef RT_H
#define RT_H

#include "types.h"

#define DO_RT_CPUS 0xF000
#define DO_MAIN_PRIO 5
#define DO_AO_PRIO 0
#define DO_AI_PRIO 0
#define DO_MAX_CHANNELS 32
#define DO_MIN_AI_BUFSZ 100
#define DO_SAMP_NS_DEFAULT ((RTIME)(0.25 * 1e6))

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Set up the basic RT machinery and launch AI/AO threads.
 */
int rtdo_init();
void rtdo_exit();

/**
 * @brief Set up an inactive AI/AO channel. The channel handle is deposited in dchan->handle.
 * @param dchan Pointer to the channel to add. The channel is not copied in order
 *  to allow asynchronous changes to the channel setup.
 * @param buffer_size Number of input or output samples in buffer.
 * @return 0 on success, error code on failure
 */
int rtdo_add_channel(daq_channel *dchan, int buffer_size);

/**
 * @brief Puts a channel into an active or inactive state asynchronously.
 *  Note that only one AO channel can be active at a time; however, the caller is responsible for ensuring this.
 * @param handle Channel handle returned by rtdo_create_channel
 * @param active 1 to activate, 0 to deactivate, -1 to get value without changing it
 * @return Previous value of the active flag, or -EINVAL if the channel handle is invalid
 */
int rtdo_set_channel_active(int handle, int active);

/**
 * @brief Deprecated, use rtdo_set_sampling_rate(DT, multiplier) instead.
 */
void rtdo_set_supersampling(int multiplier);

/**
 * @brief Set input sampling rate and supersampling rate asynchronously
 * @param ms_per_sample is the interval between samples reported through rtdo_get_data
 * @param acquisitions_per_sample is the number of actual data points read and averaged to produce a reported sample
 */
void rtdo_set_sampling_rate(double ms_per_sample, int acquisitions_per_sample);

/**
 * @brief Interrupts I/O, returning immediately.
 */
void rtdo_stop();

/**
 * @brief Interrupts I/O, applies pending asynchronous changes, empties the input buffer, and (re-)starts I/O.
 */
void rtdo_sync();

/**
 * @brief Set the stimulation pattern asynchronously. This method is thread-safe, as data are copied immediately.
 * @param handle
 * @param baseVal Value to assume at beginning and after end of stimulation pattern
 * @param numsteps Number of values/timesteps, excluding baseVal; a simple square wave would require numsteps=1.
 * @param values
 * @param ms Time points relative to the start of stimulation (sync) at which the output is to change.
 *  If the initial value is to be read from values[0], ms[0] would need to be 0. Otherwise, baseVal is applied
 *  until ms[0] is reached.
 * @param ms_total Total duration of stimulation, after which the output will be returned to baseVal.
 * @return 0 on success, or a positive error code on failure.
 */
int rtdo_set_stimulus(int handle, double baseVal, int numsteps, double *values, double *ms, double ms_total);

/**
 * @brief Get the next value from the specified channel, asynchronously.
 *  The function will wait for up to 100*DT for data to arrive.
 * @param handle
 * @param err Filled with a relevant error code on failure
 * @return A physical value sample
 */
double rtdo_get_data(int handle, int *err);

/**
 * @brief Immediately read a single value from the specified channel.
 *  It is an error to call this function while realtime input is being collected,
 *  use rtdo_get_data to read asynchronously instead.
 * @param handle
 * @param err Filled with a relevant error code on failure
 * @return A physical value sample
 */
double rtdo_read_now(int handle, int *err);

/**
 * @brief Immediately write a single value to the specified output channel.
 *  It is an error to call this function while realtime output is being generated,
 *  use rtdo_set_stimulus to write asynchronously instead.
 * @param handle
 * @param value The physical value to write
 * @return 0 on success, error code on failure
 */
int rtdo_write_now(int handle, double value);


long rtdo_thread_create(void *(*fn)(void *), void *arg, int stacksize);
void rtdo_thread_join(long thread);

#ifdef __cplusplus
}
#endif

#endif // RT_H
