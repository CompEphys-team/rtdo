/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2015-11-18

--------------------------------------------------------------------------*/

#ifndef COMEDISOFT_H
#define COMEDISOFT_H

#include <comedi.h>
#include "options.h"

enum {
    DOE_OPEN_DEV = 1,
    DOE_FIND_SUBDEV,
    DOE_LOAD_CALIBRATION,
    DOE_LOAD_LIBRARY,
    DOE_LOAD_FUNC,
    DOE_MEMORY
};

typedef struct rtdo_converter_struct rtdo_converter_type;

int rtdo_converter_init(char *calibration_file);
void rtdo_converter_exit();

int rtdo_converter_create(char *device, rtdo_channel_options *chan);

double rtdo_convert_to_physical(lsampl_t, rtdo_converter_type *);
lsampl_t rtdo_convert_from_physical(double, rtdo_converter_type *);

#endif // COMEDISOFT_H
