#include "comediconverter.h"
#include <comedilib.h>
#include <QString>
#include <assert.h>
#include <iostream>

struct mycomedi_polynomial_t : public comedi_polynomial_t {};
struct mycomedi_range : public comedi_range {};

ComediConverter::ComediConverter(const ChnData &chnp, const DAQData *devp, bool in) :
    isInChn(in),
    has_cal(false),
    polynomial(new mycomedi_polynomial_t),
    range(new mycomedi_range),
    gain(chnp.gain),
    offset(chnp.offset)
{
    comedi_t *dev = comedi_open(devp->devname().c_str());
    if ( !dev ) {
        delete polynomial;  polynomial = nullptr;
        delete range;       range = nullptr;
        throw std::runtime_error(QString("Failed to open device %1")
                                 .arg(QString::fromStdString(devp->devname()))
                                 .toStdString());
    }
    try {
        int subdev = comedi_find_subdevice_by_type(dev, in ? COMEDI_SUBD_AI : COMEDI_SUBD_AO, 0);
        if ( subdev < 0 )
            throw std::runtime_error(QString("Failed to open %1 subdevice on device %2")
                                     .arg(in ? "input":"output").arg(QString::fromStdString(devp->devname()))
                                     .toStdString());

        comedi_range *rangep = comedi_get_range(dev, subdev, chnp.idx, chnp.range);
        if ( !rangep )
            throw std::runtime_error(QString("Failed to acquire data for range %1 on device %2, subd %3, channel %4")
                                     .arg(chnp.range).arg(QString::fromStdString(devp->devname())).arg(subdev).arg(chnp.idx)
                                     .toStdString());
        *((comedi_range*)range) = *rangep;
        maxdata = comedi_get_maxdata(dev, subdev, chnp.idx);
        if ( !maxdata )
            throw std::runtime_error(QString("Failed to acquire maxdata on device %1, subd %2, channel %3")
                                     .arg(QString::fromStdString(devp->devname()).arg(subdev).arg(chnp.idx))
                                     .toStdString());

        comedi_conversion_direction direction = (in ? COMEDI_TO_PHYSICAL : COMEDI_FROM_PHYSICAL);
        if ( comedi_get_subdevice_flags(dev, subdev) & SDF_SOFT_CALIBRATED ) {
            char *cf = comedi_get_default_calibration_path(dev);
            comedi_calibration_t *cal = comedi_parse_calibration_file(cf);
            if ( !cal )
                comedi_perror(QString("Warning: Calibration file %1 invalid or inaccessible")
                              .arg(cf)
                              .toStdString().c_str());
            else if ( comedi_get_softcal_converter(subdev, chnp.idx, chnp.range, direction, cal, polynomial) )
                comedi_perror(QString("Warning: Failed to load calibration from file %1 for device %2, subd %3, channel %4, range %5")
                              .arg(cf).arg(QString::fromStdString(devp->devname())).arg(subdev).arg(chnp.idx).arg(chnp.range)
                              .toStdString().c_str());
            else
                has_cal = true;
            if ( cal )
                comedi_cleanup_calibration(cal);
            free(cf);
        } else {
            if ( comedi_get_hardcal_converter(dev, subdev, chnp.idx, chnp.range, direction, polynomial) ) {
                comedi_perror(QString("Warning: Failed to load hardware calibration for device %1")
                              .arg(QString::fromStdString(devp->devname()))
                              .toStdString().c_str());
            } else {
                has_cal = true;
            }
        }
    } catch (std::runtime_error) {
        comedi_close(dev);
        delete polynomial;  polynomial = nullptr;
        delete range;       range = nullptr;
        throw;
    }
}

ComediConverter::~ComediConverter()
{
    delete polynomial;
    delete range;
}

double ComediConverter::toPhys(lsampl_t sample) const
{
    assert(isInChn);
    double raw;
    if ( has_cal )
        raw = comedi_to_physical(sample, polynomial);
    else
        raw = comedi_to_phys(sample, range, maxdata);
    return (raw * gain) + offset;
}

lsampl_t ComediConverter::toSamp(double phys) const
{
    assert(!isInChn);
    phys = (phys - offset) / gain;
    if ( phys > range->max || phys < range->min ) {
        std::cerr << "Warning: Value out of range: " << phys << " not in ["
                  << range->min << ", " << range->max << "]" << std::endl;
    }
    if ( has_cal )
        return comedi_from_physical(phys, polynomial);
    else
        return comedi_from_phys(phys, range, maxdata);
}
