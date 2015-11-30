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
#include "rtdo_types.h"

enum {
    DOE_OPEN_DEV = 1,
    DOE_LOAD_CALIBRATION,
    DOE_LOAD_LIBRARY,
    DOE_LOAD_FUNC,
    DOE_MEMORY
};

int rtdo_converter_init(const char *calibration_file);
void rtdo_converter_exit();

int rtdo_converter_create(char *device, rtdo_channel *chan, double conversion_factor, double offset);

double rtdo_convert_to_physical(lsampl_t, rtdo_converter *);
lsampl_t rtdo_convert_from_physical(double, rtdo_converter *);

#endif // COMEDISOFT_H
