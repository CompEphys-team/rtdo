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
#include <dlfcn.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#include "softrtdaq.h"

#define DLSYM(fn, name) \
    if ( !( *(void**)(&fn) = dlsym(libcomedi, name)) && (dlerr=dlerror())) {\
        perror(dlerr);\
        return EINVAL;\
    }

struct rtdo_converter_struct {
    lsampl_t maxdata;
    comedi_range range;
    comedi_calibration_t *cal;
    comedi_polynomial_t polynomial;
};

static void *libcomedi=0;
static char *libfile = "libcomedi.so";
static comedi_t *dev=0;
static char *devfile = "/dev/comedi0";
static comedi_calibration_t *calibration=0;

static comedi_t* (*daq_open)(const char *);
static int (*daq_close)(comedi_t *);
static int (*find_subdevice_by_type)(comedi_t *, int, unsigned int);
static comedi_range* (*get_range)(comedi_t *, unsigned int, unsigned int, unsigned int);
static int (*get_n_ranges)(comedi_t *, unsigned int, unsigned int);
static int (*get_n_channels)(comedi_t *, unsigned int);
static int (*get_subdevice_flags)(comedi_t *, unsigned int);
static lsampl_t (*get_maxdata)(comedi_t *, unsigned int, unsigned int);
static comedi_calibration_t *(*parse_calibration_file)(const char *);
static void (*cleanup_calibration)(comedi_calibration_t *);
static int (*get_softcal_converter)(
    unsigned, unsigned, unsigned,
    enum comedi_conversion_direction,
    const comedi_calibration_t *, comedi_polynomial_t *);
static double (*to_physical)(lsampl_t, const comedi_polynomial_t *);
static double (*to_phys)(lsampl_t, comedi_range *, lsampl_t);
static lsampl_t (*from_physical)(double, const comedi_polynomial_t *);
static lsampl_t (*from_phys)(double, comedi_range *, lsampl_t);

int daq_load_lib(const char *libfile_) {
    if ( libcomedi )
        daq_unload_lib();
    if ( ! libfile_ )
        libfile_ = libfile;
    else
        libfile = strdup(libfile_);
    dlerror();
    if ( ! (libcomedi = dlopen(libfile_, RTLD_NOW | RTLD_DEEPBIND)) ) {
        perror(dlerror());
        return ENOENT;
    }
    
    char *dlerr;
    DLSYM(daq_open, "comedi_open");
    DLSYM(daq_close, "comedi_close");
    DLSYM(find_subdevice_by_type, "comedi_find_subdevice_by_type");
    DLSYM(get_range, "comedi_get_range");
    DLSYM(get_n_ranges, "comedi_get_n_ranges");
    DLSYM(get_n_channels, "comedi_get_n_channels");
    DLSYM(get_subdevice_flags, "comedi_get_subdevice_flags");
    DLSYM(get_maxdata, "comedi_get_maxdata");
    DLSYM(parse_calibration_file, "comedi_parse_calibration_file");
    DLSYM(get_softcal_converter, "comedi_get_softcal_converter");
    DLSYM(cleanup_calibration, "comedi_cleanup_calibration");
    DLSYM(to_physical, "comedi_to_physical");
    DLSYM(to_phys, "comedi_to_phys");
    DLSYM(from_physical, "comedi_from_physical");
    DLSYM(from_phys, "comedi_from_phys");
    
    return 0;
}

void daq_unload_lib() {
    if ( libcomedi )
        dlclose(libcomedi);
    libcomedi = 0;
}

int daq_open_device(const char *device) {
    int err;
    if ( ! libcomedi && (err = daq_load_lib(NULL)) )
        return err;
    if ( dev )
        daq_close_device();

    if ( !device )
        device = devfile;
    else
        devfile = strdup(device);

    if ( !(dev = daq_open(device)) ) {
        perror("Device not found");
        return ENODEV;
    }
    return 0;
}

void daq_close_device() {
    if ( dev )
        daq_close(dev);
    dev = 0;
}

int daq_load_calibration(const char *calibration_file) {
    if ( !calibration_file
         || access(calibration_file, F_OK) ) {
        perror("Calibration file not found");
        return ENOENT;
    }
    if ( !(calibration = parse_calibration_file(calibration_file)) ) {
        perror("Calibration file invalid");
        return EINVAL;
    }
    return 0;
}

void daq_unload_calibration() {
    if ( calibration )
        cleanup_calibration(calibration);
}

int daq_get_subdevice(const enum comedi_subdevice_type type, const unsigned int subd_offset) {
    return find_subdevice_by_type(dev, type, subd_offset);
}

int daq_get_n_channels(const unsigned int subdevice) {
    return get_n_channels(dev, subdevice);
}

int daq_get_n_ranges(const unsigned int subdevice, const int channel) {
    return get_n_ranges(dev, subdevice, channel);
}

daq_range daq_get_range(const unsigned int subdevice, const unsigned int channel, const unsigned int range) {
    daq_range r;
    comedi_range *cr = get_range(dev, subdevice, channel, range);
    r.min = cr->min;
    r.max = cr->max;
    return r;
}

int daq_get_subdevice_flags(const unsigned int subdev) {
    return get_subdevice_flags(dev, subdev);
}

int daq_create_converter(daq_channel *chan) {
    int err;
    if ( !dev && (err = daq_open_device(NULL)) )
        return err;

    if ( chan->converter ) {
        free(chan->converter);
        chan->converter = 0;
    }

    enum comedi_conversion_direction direction;
    rtdo_converter *converter;
    comedi_range *range_p;

    if ( chan->type == COMEDI_SUBD_AO ) {
        direction = COMEDI_FROM_PHYSICAL;
    } else if ( chan->type == COMEDI_SUBD_AI ) {
        direction = COMEDI_TO_PHYSICAL;
    }

    if ( !(converter = malloc(sizeof(*converter))) ) {
        perror ("Out of memory");
        return ENOMEM;
    }

    range_p = get_range(dev, chan->subdevice, chan->channel, chan->range);
    memcpy(&converter->range, range_p, sizeof(*range_p)); // Need to copy because comedi_close discards ranges.

    converter->maxdata = get_maxdata(dev, chan->subdevice, chan->channel);

    if ( calibration ) {
        if ( get_softcal_converter(chan->subdevice, chan->channel, chan->range,
                                   direction, calibration, &(converter->polynomial)) ) {
            perror("Loading softcalibrated converter failed");
            return EPERM;
        }
        converter->cal = calibration;
    } else {
        converter->cal = 0;
    }

    chan->converter = converter;
    return 0;
}

double daq_convert_to_physical(lsampl_t in, daq_channel *chan) {
    double out;
    if ( chan->converter->cal )
        out = to_physical(in, &(chan->converter->polynomial));
    else
        out = to_phys(in, &(chan->converter->range), chan->converter->maxdata);
    return (out * chan->gain) - chan->offset;
}


lsampl_t daq_convert_from_physical(double out, daq_channel *chan) {
    double Vcmd = (out - chan->offset) / chan->gain;
    if ( Vcmd > chan->converter->range.max || Vcmd < chan->converter->range.min ) {
        fprintf(stderr, "Warning: Value out of range: %f not in [%f, %f]\n", Vcmd,
                chan->converter->range.min, chan->converter->range.max);
    }
    if ( chan->converter->cal )
        return from_physical(Vcmd, &(chan->converter->polynomial));
    else
        return from_phys(Vcmd, &(chan->converter->range), chan->converter->maxdata);
}
