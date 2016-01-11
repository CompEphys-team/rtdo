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

#define SIMDIR "/home/felix/projects/rtdo/simulation"
#define INSTANCEDIR "/home/felix/projects/rtdo/models"
#define DO_DEVICE_BASE "/dev/comedi"
#define DO_MAX_DEVICES 32

#include "types.h"

#ifdef __cplusplus
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
