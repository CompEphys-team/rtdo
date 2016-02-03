/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-27

--------------------------------------------------------------------------*/
#ifndef RC_WRAPPER_H
#define RC_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <comedi.h>

struct comedi_t_struct *RC_comedi_open(const char *filename);
int RC_comedi_close(void *dev);
int RC_comedi_data_read_hint(void *dev, unsigned int subdev, unsigned int chan, unsigned int range, unsigned int aref);
int RC_comedi_data_read(void *dev, unsigned int subdev, unsigned int chan, unsigned int range, unsigned int aref, lsampl_t *data);
int RC_comedi_data_write(void *dev, unsigned int subdev, unsigned int chan, unsigned int range, unsigned int aref, lsampl_t data);

#ifdef __cplusplus
}
#endif

#endif // RC_WRAPPER_H
