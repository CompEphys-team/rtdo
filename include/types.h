/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2015-12-03

--------------------------------------------------------------------------*/
#ifndef TYPES_H
#define TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

#include <rtai_sem.h>
#include <rtai_mbx.h>
#include <comedi.h>

typedef struct rtdo_converter_struct rtdo_converter;

typedef struct {
    double min;
    double max;
} daq_range;

typedef struct {
    enum comedi_subdevice_type type;
    int subdevice;
    unsigned short channel;
    unsigned short range;
    unsigned short aref;
    double gain;
    double offset;
    rtdo_converter *converter;
    int handle;
} daq_channel;

#define DAQ_CHANNEL_INIT {COMEDI_SUBD_UNUSED, 0, 0, 0, AREF_GROUND, 1.0, 0.0, NULL, 0}

typedef struct {
    daq_channel *chan;
    int active;

    int bufsz;
    lsampl_t *buffer;
    RTIME *t;
    int numsteps;

    MBX *mbx;
} rtdo_channel;

typedef struct {
    char running;
    char exit;
    char dirty;

    SEM *load;
    SEM *presync;
    SEM *sync;

    long thread;

    rtdo_channel **chans;
    int *num_chans;
    void *dev;

    RTIME samp_ticks;
    int supersampling;
} rtdo_thread_runinfo;

#ifdef __cplusplus
}
#endif

#endif // TYPES_H
