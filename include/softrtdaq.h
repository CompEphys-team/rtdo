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

int daq_setup_channel(daq_channel *chan); //!< Sets up a channel, establishing access to its device, subdevice and converter.
void daq_create_converter(daq_channel *chan);  //!< (Re-)loads the converter for a channel on the basis of its current setup.

void daq_exit();  //!< Clean up, closing all open devices. Does not deallocate any channels.

double daq_convert_to_physical(lsampl_t in, daq_channel *chan);
lsampl_t daq_convert_from_physical(double out, daq_channel *chan);

void daq_create_channel(daq_channel *c); //!< Allocate a new, zero-initialised channel
void daq_copy_channel(daq_channel *dest, daq_channel *src); //!< Copies data from src to dest. It is an error to pass pointers to uninitialised channels, but perfectly permissible to copy over a pre-used channel.
void daq_delete_channel(daq_channel *c); //!< Deallocate a channel

#ifdef __cplusplus
}
#endif

#endif // SOFTRTDAQ_H
