#ifndef RTDO_TYPES_H
#define RTDO_TYPES_H

#include <rtai_sem.h>
#include <rtai_mbx.h>
#include <comedi.h>

enum rtdo_channel_type {
    DO_CHANNEL_AI,
    DO_CHANNEL_AO
};

typedef struct rtdo_converter_struct rtdo_converter;

typedef struct {
    enum rtdo_channel_type type;
    int subdevice;
    unsigned short channel;
    unsigned short range;
    unsigned short aref;
    rtdo_converter *converter;
    int active;

    int bufsz;
    lsampl_t *buffer;
    RTIME *t;
    int numsteps;

    MBX *mbx;
} rtdo_channel;

typedef struct {
    char running;
    char dirty;
    SEM *load;
    SEM *presync;
    long thread;
} rtdo_thread_runinfo;

#endif // RTDO_TYPES_H
