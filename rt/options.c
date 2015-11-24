/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2015-11-23

--------------------------------------------------------------------------*/
/* NOTE : This file only included as a stop-gap measure until a GUI is implemented. */

#include <comedi.h>
#include "options.h"

rtdo_daq_options daqopts = {
    "/dev/comedi0",
    "/home/felix/projects/rtdo/ni6251.calibrate"
};

rtdo_channel_options inchan_vclamp_Im = {
    DO_CHANNEL_AI,
    0,
    0,
    0,
    AREF_DIFF,
    100 /* nA/V */
};

rtdo_channel_options outchan_vclamp_Vc = {
    DO_CHANNEL_AO,
    0,
    0,
    0,
    AREF_GROUND,
    20
};
