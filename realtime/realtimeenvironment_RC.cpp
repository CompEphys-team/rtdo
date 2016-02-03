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
#include "RC_wrapper.h"
#include <rtai_mbx.h>

RealtimeEnvironment::Impl::Impl() :
    task(rt_task_init(nam2num("ENVIRO"), RealtimeEnvironment_Priority, 0, 0), &rt_thread_delete)
{
    // Set up RT
    if ( !task.get() ) {
        throw RealtimeException(RealtimeException::Setup, "rt_task_init");
    }
    rt_set_oneshot_mode();
    if ( !rt_is_hard_timer_running() )
        start_rt_timer(0);

    // Set up sync semaphore
    sync.reset(new RealtimeConditionVariable);

    // Test mailbox function
    MBX *mbx = rt_typed_mbx_init(nam2num("TESTBX"), sizeof(int), FIFO_Q);
    int msg = 1;
    rt_mbx_send(mbx, &msg, sizeof(int));
    msg = 0;
    rt_mbx_receive(mbx, &msg, sizeof(int));
    rt_mbx_delete(mbx);
    if ( !msg ) {
        throw RealtimeException(RealtimeException::MailboxTest);
    }

    // Launch I/O threads
    in.reset(new AnalogThread(true, sync));
    out.reset(new AnalogThread(false, sync));

    rt_make_soft_real_time();
}

RealtimeEnvironment::Impl::~Impl()
{
    closeDevicesRC();
    closeDevicesSC();
}

struct comedi_t_struct *RealtimeEnvironment::Impl::getDeviceRC(int deviceno)
{
    try {
        return devRC.at(deviceno);
    } catch (std::out_of_range &e) {
        std::string devname(RealtimeEnvironment_DeviceBase);
        devname += std::to_string(deviceno);
        struct comedi_t_struct *d = RC_comedi_open(devname.c_str());
        if ( !d ) {
            throw RealtimeException(RealtimeException::RuntimeFunc, std::string("RC_comedi_open(")+devname+")", 0);
        }
        return devRC[deviceno] = (struct comedi_t_struct *) d;
    }
}

void RealtimeEnvironment::Impl::closeDevicesRC()
{
    for ( std::map<int, struct comedi_t_struct *>::iterator it = devRC.begin(); it != devRC.end(); ++it ) {
        RC_comedi_close(it->second);
    }
}

#endif
