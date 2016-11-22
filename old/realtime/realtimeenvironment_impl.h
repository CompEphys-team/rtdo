/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-26

--------------------------------------------------------------------------*/
#ifndef REALTIMEENVIRONMENT_IMPL_H
#define REALTIMEENVIRONMENT_IMPL_H

extern "C" {
#include <rtai_lxrt.h>
}
#include "realtimeenvironment.h"
#include "realtimeconditionvariable.h"
#include "analogthread.h"
#include <map>

#ifndef RealtimeEnvironment_Priority
#define RealtimeEnvironment_Priority 50
#endif

#ifndef RealtimeEnvironment_DeviceBase
#define RealtimeEnvironment_DeviceBase "/dev/comedi"
#endif

class RealtimeEnvironment::Impl
{
public:
    Impl();
    ~Impl();

    struct comedi_t_struct *getDeviceRC(int deviceno);
    struct comedi_t_struct *getDeviceSC(int deviceno);

    void closeDevicesRC();
    void closeDevicesSC();

    std::string getDeviceName(int deviceno);
    unsigned int getSubdevice(int deviceno, Channel::Direction type);

    std::unique_ptr<RT_TASK, int (*)(RT_TASK *)> task;

    std::shared_ptr<RealtimeConditionVariable> sync;

    std::unique_ptr<AnalogThread> in;
    std::unique_ptr<AnalogThread> out;

    std::map<int, struct comedi_t_struct *>devSC;
    std::map<int, struct comedi_t_struct *>devRC;

    int nSyncs;
    int supersamplingRecommend;

    RTIME last_sync_ns;
};

#endif // REALTIMEENVIRONMENT_IMPL_H
