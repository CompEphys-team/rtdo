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

#define DAQCHAN_NAMELEN 32

typedef struct {
    int handle;
    char name[DAQCHAN_NAMELEN+1];

    unsigned int deviceno;
    struct comedi_t_struct *device;
    enum comedi_subdevice_type type;
    int subdevice;

    unsigned short channel;
    unsigned short range;
    unsigned short aref;

    double gain;
    double offset;

    char *softcal_file;
    struct daq_converter *converter;
} daq_channel;

typedef struct {
    daq_channel *chan;
    void *dev;
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

    RTIME samp_ticks;
    int supersampling;
} rtdo_thread_runinfo;

#ifdef __cplusplus
}
#endif

#endif // TYPES_H
