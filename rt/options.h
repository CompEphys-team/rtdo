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

enum rtdo_channel_type {
    DO_CHANNEL_AI,
    DO_CHANNEL_AO
};

typedef struct {
    char *device;
    char *calibration_file;
} rtdo_daq_options;

typedef struct {
    enum rtdo_channel_type type;
    unsigned short subdevice_offset;

    unsigned short channel;
    unsigned short range; // Check comedi_board_info for available ranges
    unsigned short reference; // One of: AREF_GROUND, AREF_COMMON, AREF_DIFF, AREF_OTHER

    double conversion_factor; // in [mV | nA]/V
} rtdo_channel_options;

extern rtdo_daq_options daqopts;

extern rtdo_channel_options inchan_vclamp_Im;
extern rtdo_channel_options outchan_vclamp_Vc;

#endif // OPTIONS_H
