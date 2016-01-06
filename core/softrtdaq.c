/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2015-12-03

--------------------------------------------------------------------------*/
#include <comedilib.h>
#include <unistd.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#include "globals.h"
#include "softrtdaq.h"

struct daq_converter {
    char has_cal;
    comedi_polynomial_t polynomial;
    comedi_range *range;
    lsampl_t maxdata;
};

static comedi_t *devices[DO_MAX_DEVICES] = {0};

void daq_create_converter(daq_channel *chan) {
    enum comedi_conversion_direction direction;
    if ( chan->type == COMEDI_SUBD_AO )
        direction = COMEDI_FROM_PHYSICAL;
    else
        direction = COMEDI_TO_PHYSICAL;

    if ( !chan->converter && !(chan->converter = malloc(sizeof(*chan->converter))) )
        exit(ENOMEM);
    chan->converter->has_cal = 0;
    if ( comedi_get_subdevice_flags(chan->device, chan->subdevice) & SDF_SOFT_CALIBRATED ) {
        char *cf = chan->softcal_file;
        int free_cf = 0;
        if ( !cf ) {
            cf = comedi_get_default_calibration_path(chan->device);
            free_cf = 1;
        }
        comedi_calibration_t *cal = comedi_parse_calibration_file(cf);
        if ( !cal || comedi_get_softcal_converter(chan->subdevice, chan->channel, chan->range,
                                                  direction, cal, &chan->converter->polynomial) ) {
                comedi_perror("Warning: Loading softcalibrated converter failed");
        } else {
            chan->converter->has_cal = 1;
        }
        if ( cal )
            comedi_cleanup_calibration(cal);
        if ( free_cf )
            free(cf);
    } else {
        if ( comedi_get_hardcal_converter(chan->device, chan->subdevice, chan->channel,
                                          chan->range, direction, &chan->converter->polynomial) ) {
            comedi_perror("Warning: Loading hardware calibration failed");
        } else {
            chan->converter->has_cal = 1;
        }
    }

    chan->converter->range = comedi_get_range(chan->device, chan->subdevice, chan->channel, chan->range);
    chan->converter->maxdata = comedi_get_maxdata(chan->device, chan->subdevice, chan->channel);
}

double daq_convert_to_physical(lsampl_t in, daq_channel *chan) {
    double out;
    if ( chan->converter->has_cal )
        out = comedi_to_physical(in, &chan->converter->polynomial);
    else
        out = comedi_to_phys(in, chan->converter->range, chan->converter->maxdata);
    return (out * chan->gain) - chan->offset;
}


lsampl_t daq_convert_from_physical(double out, daq_channel *chan) {
    double Vcmd = (out - chan->offset) / chan->gain;
    if ( Vcmd > chan->converter->range->max || Vcmd < chan->converter->range->min ) {
        fprintf(stderr, "Warning: Value out of range: %f not in [%f, %f]\n", Vcmd,
                chan->converter->range->min, chan->converter->range->max);
    }
    if ( chan->converter->has_cal )
        return comedi_from_physical(Vcmd, &chan->converter->polynomial);
    else
        return comedi_from_phys(Vcmd, chan->converter->range, chan->converter->maxdata);
}

int daq_open_device(unsigned int deviceno, struct comedi_t_struct **device) {
    if ( deviceno >= DO_MAX_DEVICES ) {
        return EINVAL;
    }
    if ( !devices[deviceno] ) {
        char buf[strlen(DO_DEVICE_BASE) + 5];
        sprintf(buf, "%s%d", DO_DEVICE_BASE, deviceno);
        if ( !(devices[deviceno] = comedi_open(buf)) ) {
            return ENODEV;
        }
    }
    *device = devices[deviceno];
    return 0;
}

int daq_setup_channel(daq_channel *chan) {
    int ret = daq_open_device(chan->deviceno, &chan->device);
    if ( ret ) {
        if ( ret == EINVAL ) fprintf(stderr, "Error: Device number out of range\n");
        if ( ret == ENODEV ) comedi_perror("Failed to open device");
        return ret;
    }
    if ( (chan->subdevice = comedi_find_subdevice_by_type(chan->device, chan->type, 0)) == -1 ) {
        comedi_perror("Subdevice not found");
        return ENODEV;
    }
    daq_create_converter(chan);
    return 0;
}

void daq_exit() {
    int i;
    for ( i = 0; i < DO_MAX_DEVICES; i++ ) {
        if ( devices[i] ) {
            comedi_close(devices[i]);
            devices[i] = 0;
        }
    }
}

void daq_create_channel(daq_channel *c) {
    memset(c, 0, sizeof(daq_channel));
    if ( !(c->converter = calloc(1, sizeof(struct daq_converter))) )
        exit(ENOMEM);
    c->type = COMEDI_SUBD_AI;
    c->gain = 1.0;
}

void daq_copy_channel(daq_channel *dest, daq_channel *src) {
    if ( !dest->converter )
        daq_create_channel(dest);
    struct daq_converter *tmp = dest->converter;
    memcpy(dest, src, sizeof(daq_channel));
    if ( src->converter ) {
        memcpy(tmp, src->converter, sizeof(struct daq_converter));
        dest->converter = tmp;
    }
}

void daq_delete_channel(daq_channel *c) {
    if ( c->converter )
        free(c->converter);
    c->converter = 0;
}

void daq_set_channel_name(daq_channel *c, const char *name) {
    strncpy(c->name, name, DAQCHAN_NAMELEN);
}
