/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-26

--------------------------------------------------------------------------*/
#ifndef CHANNEL_IMPL_H
#define CHANNEL_IMPL_H

#include "channel.h"
#include "realtimequeue.h"

class Channel::Impl
{
public:
    Impl(Channel::Direction type, int deviceno, unsigned int channel, unsigned int range, Channel::Aref aref);

    Impl(const Impl&);
    ~Impl();

    static unsigned int aref(Channel::Aref r);
    static Channel::Aref aref(unsigned int r);

    bool setAref(Channel *_this, Channel::Aref aref, bool doSanitise = true);

    enum DeviceProperty { pDevice, pSubdevice, pChannel, pRange, pAref };
    void sanitise(Channel *_this, DeviceProperty p);


    int _deviceno;
    struct comedi_t_struct *_deviceRC;
    struct comedi_t_struct *_deviceSC;
    unsigned int _subdevice;
    unsigned int _channel;
    unsigned int _range;
    unsigned int _aref;
    double _gain;
    double _offset;

    int offsetSrc;

    RealtimeQueue<lsampl_t> q;

    class Converter;
    std::unique_ptr<Converter> converter;
};

#endif // CHANNEL_IMPL_H
