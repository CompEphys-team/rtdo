/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-26

--------------------------------------------------------------------------*/
#ifdef CONFIG_RT

#include "channel_impl.h"
#include <comedilib.h>
#include <rtai_mbx.h>
#include <iostream>
#include "realtimeenvironment.h"
#include "config.h"

// ---------------------------------------------- Converter implementation ---------------------------
class Channel::Impl::Converter
{
public:
    bool has_cal;
    comedi_polynomial_t polynomial;
    comedi_range *range;
    lsampl_t maxdata;

    Converter(Channel::Direction type, Channel::Impl *chan) :
        has_cal(false),
        range(comedi_get_range(chan->_deviceSC, chan->_subdevice, chan->_channel, chan->_range)),
        maxdata(comedi_get_maxdata(chan->_deviceSC, chan->_subdevice, chan->_channel))
    {
        comedi_conversion_direction direction = (type == Channel::AnalogIn ? COMEDI_TO_PHYSICAL : COMEDI_FROM_PHYSICAL);
        if ( comedi_get_subdevice_flags(chan->_deviceSC, chan->_subdevice) & SDF_SOFT_CALIBRATED ) {
            char *cf = comedi_get_default_calibration_path(chan->_deviceSC);
            comedi_calibration_t *cal = comedi_parse_calibration_file(cf);
            if ( !cal || comedi_get_softcal_converter(chan->_subdevice, chan->_channel, chan->_range,
                                                      direction, cal, &polynomial) ) {
                    comedi_perror("Warning: Loading softcalibrated converter failed");
            } else {
                has_cal = true;
            }
            if ( cal )
                comedi_cleanup_calibration(cal);
            free(cf);
        } else {
            if ( comedi_get_hardcal_converter(chan->_deviceSC, chan->_subdevice, chan->_channel,
                                              chan->_range, direction, &polynomial) ) {
                comedi_perror("Warning: Loading hardware calibration failed");
            } else {
                has_cal = true;
            }
        }
    }

    ~Converter() = default;
    Converter(const Converter&) = default;
};


// ------------------------------------- Channel implementation --------------------------------------
Channel::Impl::Impl(Channel::Direction type, int deviceno, unsigned int channel, unsigned int range, Channel::Aref aref) :
    _deviceno(deviceno),
    _deviceRC(RealtimeEnvironment::env().getDevice(deviceno, true)),
    _deviceSC(RealtimeEnvironment::env().getDevice(deviceno, false)),
    _subdevice(RealtimeEnvironment::env().getSubdevice(deviceno, type)),
    _channel(channel),
    _range(range),
    _aref(Impl::aref(aref)),
    _gain(1.0),
    _offset(0.0),
    offsetSrc(0),
    mbx(rt_typed_mbx_init(0, Channel_MailboxSize * sizeof(lsampl_t), PRIO_Q), &rt_mbx_delete),
    converter(new Converter(type, this))
{}

Channel::Impl::Impl(const Impl &other) :
    _deviceno(other._deviceno),
    _deviceRC(other._deviceRC),
    _deviceSC(other._deviceSC),
    _subdevice(other._subdevice),
    _channel(other._channel),
    _range(other._range),
    _aref(other._aref),
    _gain(other._gain),
    _offset(other._offset),
    offsetSrc(other.offsetSrc),
    mbx(other.mbx),
    converter(new Converter(*other.converter))
{}

Channel::Impl::~Impl() {}

unsigned int Channel::Impl::aref(Channel::Aref r)
{
    switch ( r ) {
    case Channel::Ground: return AREF_GROUND;
    case Channel::Common: return AREF_COMMON;
    case Channel::Diff:   return AREF_DIFF;
    case Channel::Other:  return AREF_OTHER;
    default:              return 0;
    }
}

Channel::Aref Channel::Impl::aref(unsigned int r)
{
    switch ( r ) {
    case AREF_GROUND: return Channel::Ground;
    case AREF_COMMON: return Channel::Common;
    case AREF_DIFF:   return Channel::Diff;
    default:
    case AREF_OTHER:  return Channel::Other;
    }
}

bool Channel::Impl::setAref(Channel *_this, Channel::Aref aref, bool doSanitise)
{
    if ( !_this->hasAref(aref) )
        return false;
    unsigned int r = Impl::aref(aref);
    if ( _aref == r )
        return true;
    _aref = r;
    if ( doSanitise )
        sanitise(_this, pAref);
    return true;
}

void Channel::Impl::sanitise(Channel *_this, DeviceProperty p)
{
    bool brk = true;
    switch ( p ) {
    case pDevice:
        if ( _this->setDirection(_this->_type) || // Find subdevice for previous type on new device
             _this->setDirection(_this->_type == Channel::AnalogIn ? Channel::AnalogOut : Channel::AnalogIn) ) {
            break;
        } else {
            throw RealtimeException(RealtimeException::RuntimeMsg, "No valid subdevices found");
        }
    case pSubdevice:
        if ( !_this->hasAref(aref(_aref)) ) {
            brk = false;
            if ( !setAref(_this, Channel::Common, false) &&
                 !setAref(_this, Channel::Diff, false) &&
                 !setAref(_this, Channel::Ground, false) &&
                 !setAref(_this, Channel::Other, false) ) {
                throw RealtimeException(RealtimeException::RuntimeMsg, "No valid analog reference found");
            }
        }
        if ( _channel >= _this->numChannels() ) {
            brk = true;
            if ( !_this->setChannel(0) )
                throw RealtimeException(RealtimeException::RuntimeMsg, "No valid channels found");
        }
        if ( brk )
            break;
    case pChannel:
        if ( !_this->hasRange(_range) ) {
            if ( !_this->setRange(0) )
                throw RealtimeException(RealtimeException::RuntimeMsg, "No valid ranges found");
            break;
        }
    case pRange:
    case pAref:
        converter.reset(new Converter(_this->_type, this));
    }
}


// ------------------------------ Public Channel API ---------------------------------------------
lsampl_t Channel::convert(double voltage) const
{
    double Vcmd = (voltage - pImpl->_offset) / pImpl->_gain;
    if ( Vcmd > pImpl->converter->range->max || Vcmd < pImpl->converter->range->min ) {
        std::cerr << "Warning: Value out of range: " << Vcmd << " not in ["
                  << pImpl->converter->range->min << ", " << pImpl->converter->range->max << "]" << std::endl;
    }
    if ( pImpl->converter->has_cal )
        return comedi_from_physical(Vcmd, &pImpl->converter->polynomial);
    else
        return comedi_from_phys(Vcmd, pImpl->converter->range, pImpl->converter->maxdata);
}

double Channel::convert(lsampl_t sample) const
{
    double raw;
    if ( pImpl->converter->has_cal )
        raw = comedi_to_physical(sample, &pImpl->converter->polynomial);
    else
        raw = comedi_to_phys(sample, pImpl->converter->range, pImpl->converter->maxdata);
    return (raw * pImpl->_gain) + pImpl->_offset;
}

bool Channel::hasDirection(Channel::Direction type) const
{
    if ( type != Channel::AnalogIn && type != Channel::AnalogOut )
        return false;
    return ( 0 <= comedi_find_subdevice_by_type(pImpl->_deviceSC,
                                                type == Channel::AnalogIn ? COMEDI_SUBD_AI : COMEDI_SUBD_AO, 0) );
}

unsigned int Channel::numChannels() const
{
    int n = comedi_get_n_channels(pImpl->_deviceSC, pImpl->_subdevice);
    if ( n < 0 )
        throw RealtimeException(RealtimeException::RuntimeFunc, "comedi_get_n_channels", n);
    return (unsigned int) n;
}

bool Channel::hasRange(unsigned int range) const
{
    int r = comedi_get_n_ranges(pImpl->_deviceSC, pImpl->_subdevice, pImpl->_channel);
    if ( r < 0)
        throw RealtimeException(RealtimeException::RuntimeFunc, "comedi_get_n_ranges", r);
    return ( range < (unsigned int) r );
}

bool Channel::hasAref(Channel::Aref aref) const
{
    int flags = comedi_get_subdevice_flags(pImpl->_deviceSC, pImpl->_subdevice);
    switch ( aref ) {
    case Channel::Common: return flags & SDF_COMMON;
    case Channel::Diff:   return flags & SDF_DIFF;
    case Channel::Ground: return flags & SDF_GROUND;
    case Channel::Other:  return flags & SDF_OTHER;
    default:          return false;
    }
}

double Channel::rangeMin(unsigned int range) const
{
    if ( !hasRange(range) )
        throw RealtimeException(RealtimeException::RuntimeMsg, "Invalid range", range);
    comedi_range *r = comedi_get_range(pImpl->_deviceSC, pImpl->_subdevice, pImpl->_channel, range);
    return r->min;
}

double Channel::rangeMax(unsigned int range) const
{
    if ( !hasRange(range) )
        throw RealtimeException(RealtimeException::RuntimeMsg, "Invalid range", range);
    comedi_range *r = comedi_get_range(pImpl->_deviceSC, pImpl->_subdevice, pImpl->_channel, range);
    return r->max;
}

std::string Channel::rangeUnit(unsigned int range) const
{
    if ( !hasRange(range) )
        throw RealtimeException(RealtimeException::RuntimeMsg, "Invalid range", range);
    comedi_range *r = comedi_get_range(pImpl->_deviceSC, pImpl->_subdevice, pImpl->_channel, range);
    switch ( r->unit ) {
    case UNIT_mA:   return "mA";
    case UNIT_volt: return "V";
    default:        return "X";
    }
}

bool Channel::setDirection(Channel::Direction type)
{
    if ( !hasDirection(type) )
        return false;
    if ( _type == type )
        return true;
    unsigned int subdev = RealtimeEnvironment::env().getSubdevice(pImpl->_deviceno, type);
    _type = type;
    pImpl->_subdevice = subdev;
    pImpl->sanitise(this, Impl::pSubdevice);
    return true;
}

bool Channel::setDevice(int deviceno)
{
    struct comedi_t_struct *sc, *rc;
    try {
        sc = RealtimeEnvironment::env().getDevice(deviceno, false);
        rc = RealtimeEnvironment::env().getDevice(deviceno, true);
    } catch (RealtimeException &e) {
        return false;
    }
    if ( sc == pImpl->_deviceSC && rc == pImpl->_deviceRC )
        return true;
    pImpl->_deviceSC = sc;
    pImpl->_deviceRC = rc;
    pImpl->sanitise(this, Impl::pDevice);
    return true;
}

bool Channel::setChannel(unsigned int channel)
{
    if ( numChannels() <= channel )
        return false;
    if ( pImpl->_channel == channel )
        return true;
    pImpl->_channel = channel;
    pImpl->sanitise(this, Impl::pChannel);
    return true;
}

bool Channel::setRange(unsigned int range)
{
    if ( !hasRange(range) )
        return false;
    if ( pImpl->_range == range )
        return true;
    pImpl->_range = range;
    pImpl->sanitise(this, Impl::pRange);
    return true;
}

bool Channel::setAref(Channel::Aref aref)
{
    return pImpl->setAref(this, aref);
}

void Channel::setOffset(double offset)
{
    pImpl->_offset = offset;
}

void Channel::setConversionFactor(double factor)
{
    pImpl->_gain = factor;
}

void Channel::setOffsetSource(int ID)
{
    pImpl->offsetSrc = ID;
}

double Channel::offset() const { return pImpl->_offset; }
double Channel::conversionFactor() const { return pImpl->_gain; }
int Channel::device() const { return pImpl->_deviceno; }
unsigned int Channel::channel() const { return pImpl->_channel; }
unsigned int Channel::range() const { return pImpl->_range; }
Channel::Aref Channel::aref() const { return Impl::aref(pImpl->_aref); }
int Channel::offsetSource() const { return pImpl->offsetSrc; }

void Channel::flush()
{
    lsampl_t buf[Channel_MailboxSize];
    while ( rt_mbx_receive_wp(&*(pImpl->mbx), &buf, Channel_MailboxSize * sizeof(lsampl_t)) < Channel_MailboxSize ) {}
}

double Channel::nextSample()
{
    if ( _type != AnalogIn )
        throw RealtimeException(RealtimeException::RuntimeMsg, "Reading samples from output channel is not supported.");

    lsampl_t sample;
    RTIME delay = 100 * nano2count(1e6 * config->io.dt);
    if ( rt_mbx_receive_timed(&*(pImpl->mbx), &sample, sizeof(lsampl_t), delay) ) {
        throw RealtimeException(RealtimeException::Timeout, "receiving data from read queue");
    }

    return convert(sample);
}

void Channel::readOffset()
{
    double o;
    if ( pImpl->offsetSrc > 0 ) {
        for ( Channel &c : config->io.channels ) {
            if ( c.ID() == pImpl->offsetSrc && c.read(o, true) ) {
                pImpl->_offset = o;
                break;
            }
        }
    }
}

#endif
