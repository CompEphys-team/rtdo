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
#include <comedilib.h>
#include "converter.h"
#include "options.h"

comedi_polynomial_t ai_poly, ao_poly;
int calibrated;
lsampl_t ai_maxdata, ao_maxdata;
comedi_range *ai_range, *ao_range;

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


double rtdo_convert_ai_sample(lsampl_t in) {
    if ( calibrated )
        return lc_to_physical(in, &ai_poly);
    else
        return lc_to_phys(in, ai_range, ai_maxdata);
}


lsampl_t rtdo_convert_ao_sample(double out) {
    double Vcmd = out / daqopts.vc_out_mV_per_V;
    if ( Vcmd > ao_range->max || Vcmd < ao_range->min ) {
        rt_printk("Warning: Desired command voltage %.1f V (%.2f mV) is outside the channel range [%.1f V, %.1f V]\n",
                  Vcmd, out, ao_range->min, ao_range->max );
    }
    if ( calibrated )
        return lc_from_physical(Vcmd, &ao_poly);
    else
        return lc_from_phys(Vcmd, ao_range, ao_maxdata);
}


int rtdo_converter_init() {
    comedi_t *dev;
    comedi_calibration_t *cal;
    int aidev, aodev, ret=0;

    dlerror();
    if ( ! (libcomedi = dlopen("libcomedi.so", RTLD_NOW | RTLD_DEEPBIND)) ) {
        printf(dlerror());
        return DOE_LOAD_LIBRARY;
    }
    char *dlerr;
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

    if ( !(dev = lc_open(daqopts.device)) )
        return DOE_OPEN_DEV;
    aidev = lc_find_subdevice_by_type(dev, COMEDI_SUBD_AI, daqopts.ai_subdev_offset);
    aodev = lc_find_subdevice_by_type(dev, COMEDI_SUBD_AO, daqopts.ao_subdev_offset);
    if ( aidev < 0 || aodev < 0 ) {
        lc_close(dev);
        return DOE_FIND_SUBDEV;
    }

    ai_range = lc_get_range(dev, aidev, daqopts.vc_in_chan, daqopts.vc_in_range);
    ai_maxdata = lc_get_maxdata(dev, aidev, daqopts.vc_in_chan);
    ao_range = lc_get_range(dev, aodev, daqopts.vc_out_chan, daqopts.vc_out_range);
    ao_maxdata = lc_get_maxdata(dev, aodev, daqopts.vc_out_chan);

    if ( !access(daqopts.calibration_file, F_OK) && (cal = lc_parse_calibration_file(daqopts.calibration_file)) ) {
        lc_get_softcal_converter(aidev, daqopts.vc_in_chan, daqopts.vc_in_range, COMEDI_TO_PHYSICAL, cal, &ai_poly);
        lc_get_softcal_converter(aodev, daqopts.vc_out_chan, daqopts.vc_out_range, COMEDI_FROM_PHYSICAL, cal, &ao_poly);
        lc_cleanup_calibration(cal);
        calibrated = 1;
        ret = 0;
    } else {
        calibrated = 0;
        ret = DOE_LOAD_CALIBRATION;
    }

    lc_close(dev);
    return ret;
}


void rtdo_converter_exit() {
    dlclose(libcomedi);
}
