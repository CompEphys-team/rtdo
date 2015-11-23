/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2015-11-23

--------------------------------------------------------------------------*/
/* NOTE : This file only included as a stop-gap measure until a GUI is implemented. */

#include "options.h"

rtdo_daq_options daqopts = {
    "/dev/comedi0",
    0, 0,

    "/home/felix/projects/rtdo/ni6251.calibrate",

    0, 0, 20,
    0, 0, AREF_DIFF,

    1, 0, 10,
    1, 0, AREF_DIFF
};
