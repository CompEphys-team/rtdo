/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-26

--------------------------------------------------------------------------*/
#ifdef CONFIG_RT

#include "realtimeenvironment_impl.h"
#include <comedilib.h>

struct comedi_t_struct *RealtimeEnvironment::Impl::getDeviceSC(int deviceno)
{
    try {
        return devSC.at(deviceno);
    } catch (std::out_of_range &e) {
        std::string devname(RealtimeEnvironment_DeviceBase);
        devname += std::to_string(deviceno);
        comedi_t *d = comedi_open(devname.c_str());
        if ( !d ) {
            throw RealtimeException(RealtimeException::RuntimeFunc, std::string("comedi_open(")+devname+")", 0);
        }
        return devSC[deviceno] = (struct comedi_t_struct *) d;
    }
}

void RealtimeEnvironment::Impl::closeDevicesSC()
{
    for ( auto &p : devSC ) {
        comedi_close(p.second);
    }
}

unsigned int RealtimeEnvironment::Impl::getSubdevice(int deviceno, Channel::Direction type)
{
    if ( type != Channel::AnalogIn && type != Channel::AnalogOut )
        return 0;
    int subdevice = comedi_find_subdevice_by_type(getDeviceSC(deviceno),
                                                  type == Channel::AnalogIn ? COMEDI_SUBD_AI : COMEDI_SUBD_AO,
                                                  0);
    if ( subdevice < 0 ) {
        throw RealtimeException(RealtimeException::RuntimeFunc, "comedi_find_subdevice", subdevice);
    }
    return (unsigned int)subdevice;
}

std::string RealtimeEnvironment::Impl::getDeviceName(int deviceno) {
    struct comedi_t_struct *d = getDeviceSC(deviceno);
    return std::string(comedi_get_board_name(d));
}

#endif
