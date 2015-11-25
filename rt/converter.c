/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2015-11-17

--------------------------------------------------------------------------*/

#include <unistd.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <rtai_lxrt.h>
#include <comedilib.h>
#include "converter.h"
#include "options.h"

struct rtdo_converter_struct {
    lsampl_t maxdata;
    comedi_range range;
    comedi_polynomial_t polynomial;
    double conversion_factor;
};

static comedi_calibration_t *calibration;

// dlsym'ed functions. Check comedilib.h for s/lc_/comedi_/ signatures.
void *libcomedi;
comedi_t* (*lc_open)(const char *);
int (*lc_close)(comedi_t *);
int (*lc_find_subdevice_by_type)(comedi_t *, int, unsigned int);
comedi_range* (*lc_get_range)(comedi_t *, unsigned int, unsigned int, unsigned int);
lsampl_t (*lc_get_maxdata)(comedi_t *, unsigned int, unsigned int);
comedi_calibration_t *(*lc_parse_calibration_file)(const char *);
void (*lc_cleanup_calibration)(comedi_calibration_t *);
int (*lc_get_softcal_converter)(
    unsigned, unsigned, unsigned,
    enum comedi_conversion_direction,
    const comedi_calibration_t *, comedi_polynomial_t *);
double (*lc_to_physical)(lsampl_t, const comedi_polynomial_t *);
double (*lc_to_phys)(lsampl_t, comedi_range *, lsampl_t);
lsampl_t (*lc_from_physical)(double, const comedi_polynomial_t *);
lsampl_t (*lc_from_phys)(double, comedi_range *, lsampl_t);

// Bind fn to dlsym(handle, name) and check for errors.
// See man 3 dlsym for a rationale for the *(void**)(&fn) = ...
#define DLSYM(fn, handle, name) \
    if ( !( *(void**)(&fn) = dlsym(handle, name)) && (dlerr=dlerror())) {\
        printf(dlerr);\
        return DOE_LOAD_FUNC;\
    }


double rtdo_convert_to_physical(lsampl_t in, rtdo_converter_type *converter) {
    double out;
    if ( calibration )
        out = lc_to_physical(in, &(converter->polynomial));
    else
        out = lc_to_phys(in, &(converter->range), converter->maxdata);
    return out * converter->conversion_factor;
}


lsampl_t rtdo_convert_from_physical(double out, rtdo_converter_type *converter) {
    double Vcmd = out / converter->conversion_factor;
    if ( Vcmd > converter->range.max || Vcmd < converter->range.min ) {
        rt_printk("Warning: Desired command voltage %.1f V (%.2f mV or nA) is outside the channel range [%.1f V, %.1f V]\n",
                  Vcmd, out, converter->range.min, converter->range.max );
    }
    if ( calibration )
        return lc_from_physical(Vcmd, &(converter->polynomial));
    else
        return lc_from_phys(Vcmd, &(converter->range), converter->maxdata);
}


int rtdo_converter_init(char *calibration_file) {
    char *dlerr;

    dlerror();
    if ( ! (libcomedi = dlopen("libcomedi.so", RTLD_NOW | RTLD_DEEPBIND)) ) {
        printf(dlerror());
        return DOE_LOAD_LIBRARY;
    }
    DLSYM(lc_open, libcomedi, "comedi_open");
    DLSYM(lc_close, libcomedi, "comedi_close");
    DLSYM(lc_find_subdevice_by_type, libcomedi, "comedi_find_subdevice_by_type");
    DLSYM(lc_get_range, libcomedi, "comedi_get_range");
    DLSYM(lc_get_maxdata, libcomedi, "comedi_get_maxdata");
    DLSYM(lc_parse_calibration_file, libcomedi, "comedi_parse_calibration_file");
    DLSYM(lc_get_softcal_converter, libcomedi, "comedi_get_softcal_converter");
    DLSYM(lc_cleanup_calibration, libcomedi, "comedi_cleanup_calibration");
    DLSYM(lc_to_physical, libcomedi, "comedi_to_physical");
    DLSYM(lc_to_phys, libcomedi, "comedi_to_phys");
    DLSYM(lc_from_physical, libcomedi, "comedi_from_physical");
    DLSYM(lc_from_phys, libcomedi, "comedi_from_phys");

    if ( calibration_file
         && !access(calibration_file, F_OK)
         && (calibration = lc_parse_calibration_file(calibration_file)) ) {
        return 0;
    } else {
        calibration = 0;
        return DOE_LOAD_CALIBRATION;
    }
}


int rtdo_converter_create(char *device, rtdo_channel_options *chan) {
    comedi_t *dev;
    int subdev;
    enum comedi_conversion_direction direction;
    enum comedi_subdevice_type subdev_type;
    rtdo_converter_type *converter;
    comedi_range *range_p;

    if ( chan->type == DO_CHANNEL_AO ) {
        direction = COMEDI_FROM_PHYSICAL;
        subdev_type = COMEDI_SUBD_AO;
    } else if ( chan->type == DO_CHANNEL_AI ) {
        direction = COMEDI_TO_PHYSICAL;
        subdev_type = COMEDI_SUBD_AI;
    }

    if ( !(dev = lc_open(device)) ) {
        return DOE_OPEN_DEV;
    }
    subdev = lc_find_subdevice_by_type(dev, subdev_type, chan->subdevice_offset);
    if ( subdev < 0 ) {
        lc_close(dev);
        return DOE_FIND_SUBDEV;
    }

    if ( ! (converter = malloc(sizeof(*converter))) ) {
        return DOE_MEMORY;
    } /* TODO: keep track of these pointers */

    range_p = lc_get_range(dev, subdev, chan->channel, chan->range);
    memcpy(&converter->range, range_p, sizeof(*range_p)); // Need to copy because comedi_close discards ranges.

    converter->maxdata = lc_get_maxdata(dev, subdev, chan->channel);
    converter->conversion_factor = chan->conversion_factor;

    if ( calibration ) {
        lc_get_softcal_converter(subdev, chan->channel, chan->range,
                                 direction, calibration, &(converter->polynomial));
    }

    lc_close(dev);
    chan->converter = (void *)converter;
    return 0;
}


void rtdo_converter_exit() {
    if ( calibration )
        lc_cleanup_calibration(calibration);
    dlclose(libcomedi);
}
