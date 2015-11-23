/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2015-11-23

--------------------------------------------------------------------------*/

#ifndef OPTIONS_H
#define OPTIONS_H

#include <comedi.h>

typedef struct {
    char *device;
    unsigned short ai_subdev_offset;
    unsigned short ao_subdev_offset;

    char *calibration_file;

    unsigned short vc_out_chan;
    unsigned short vc_out_range;
    int vc_out_mV_per_V;

    unsigned short vc_in_chan;
    unsigned short vc_in_range;
    unsigned short vc_in_ref; // One of: AREF_GROUND, AREF_COMMON, AREF_DIFF, AREF_OTHER

    unsigned short cc_out_chan;
    unsigned short cc_out_range;
    int cc_out_nA_per_V;

    unsigned short cc_in_chan;
    unsigned short cc_in_range;
    unsigned short cc_in_ref;
} rtdo_daq_options;

extern rtdo_daq_options daqopts;

#endif // OPTIONS_H
