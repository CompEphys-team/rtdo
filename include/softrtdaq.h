/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2015-12-03

--------------------------------------------------------------------------*/
#ifndef SOFTRTDAQ_H
#define SOFTRTDAQ_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

int daq_load_lib(const char *libfile);
void daq_unload_lib();

int daq_open_device(const char *device);
void daq_close_device();

int daq_load_calibration(const char *calibration_file);
void daq_unload_calibration();

int daq_get_subdevice(const enum comedi_subdevice_type type, const unsigned int subd_offset);
int daq_get_n_channels(const unsigned int subdevice);
int daq_get_n_ranges(const unsigned int subdevice, const int channel);
daq_range daq_get_range(const unsigned int subdevice, const unsigned int channel, const unsigned int range);
int daq_get_subdevice_flags(const unsigned int subdev);

int daq_create_converter(daq_channel *chan);

double daq_convert_to_physical(lsampl_t in, daq_channel *chan);
lsampl_t daq_convert_from_physical(double out, daq_channel *chan, int *err);

double daq_read_value(daq_channel *chan, int *err);

#ifdef __cplusplus
}
#endif

#endif // SOFTRTDAQ_H
