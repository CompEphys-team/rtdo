/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-27

--------------------------------------------------------------------------*/
#ifdef CONFIG_RT

#include "RC_wrapper.h"
#include "RC_rtai_comedi.h"

struct comedi_t_struct *RC_comedi_open(const char *filename)
{
    return RC_C_comedi_open(filename);
}

int RC_comedi_close(void *dev)
{
    return RC_C_comedi_close(dev);
}

int RC_comedi_data_read_hint(void *dev, unsigned int subdev, unsigned int chan, unsigned int range, unsigned int aref)
{
    return RC_C_comedi_data_read_hint(dev, subdev, chan, range, aref);
}

int RC_comedi_data_read(void *dev, unsigned int subdev, unsigned int chan, unsigned int range, unsigned int aref, lsampl_t *data)
{
    return RC_C_comedi_data_read(dev, subdev, chan, range, aref, data);
}

int RC_comedi_data_write(void *dev, unsigned int subdev, unsigned int chan, unsigned int range, unsigned int aref, lsampl_t data)
{
    return RC_C_comedi_data_write(dev, subdev, chan, range, aref, data);
}

#endif
