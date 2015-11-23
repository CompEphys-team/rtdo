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

#define DO_DEV "/dev/comedi0"
#define DO_AOCHAN 0
#define DO_AORANGE 0
#define DO_AICHAN 0
#define DO_AIRANGE 0
#define DO_COMMAND_MV_PER_V 20

#define DO_CALIBRATION_FILE "/home/felix/projects/rtdo/ni6251.calibrate"

enum {
    DOE_OPEN_DEV = 1,
    DOE_FIND_SUBDEV,
    DOE_LOAD_CALIBRATION,
    DOE_LOAD_LIBRARY,
    DOE_LOAD_FUNC
};

int rtdo_converter_init();
void rtdo_converter_exit();
double rtdo_convert_ai_sample(lsampl_t in);
lsampl_t rtdo_convert_ao_sample(double in);

#endif // COMEDISOFT_H
