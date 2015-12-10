/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2015-12-03

--------------------------------------------------------------------------*/
#include <rtai_comedi.h>
#include <rtai_sem.h>
#include <rtai_mbx.h>
#include "rt.h"

void *ao_fun(void *runinfo) {
    RT_TASK *task;
    int ret = 1, i, step, steps=0, bufsz=0;
    rtdo_channel *chan=0;
    daq_channel dchan;
    RTIME now, expected, *t=0;
    lsampl_t *buffer=0;
    rtdo_thread_runinfo *run = (rtdo_thread_runinfo *)runinfo;

    task = rt_thread_init(nam2num("AOFUN"), DO_AO_PRIO, 5000, SCHED_FIFO, DO_RT_CPUS);
    if (!task) {
        perror("AO RT setup failed");
        return (void *)EXIT_FAILURE;
    }

    mlockall(MCL_CURRENT | MCL_FUTURE);
    rt_make_hard_real_time();

    while ( !run->exit ) {
        // Load new channel config
        rt_sem_wait(run->load);
        if ( run->dirty ) {
            rt_make_soft_real_time();
            chan = 0;
            for ( i = 1; i < *(run->num_chans); i++ ) {
                if ( run->chans[i]->chan->type == COMEDI_SUBD_AO && run->chans[i]->active ) {
                    chan = run->chans[i];
                    if ( chan->numsteps > bufsz ) {
                        free(buffer);
                        free(t);
                        bufsz = chan->numsteps;
                        if ( !(buffer = malloc(bufsz*sizeof(*buffer)))
                             || !(t = malloc(bufsz*sizeof(*t))) ) {
                            rt_make_soft_real_time();
                            rt_thread_delete(task);
                            perror("AO out of memory");
                            return (void *)ENOMEM;
                        }
                    }
                    steps = chan->numsteps;
                    memcpy(buffer, chan->buffer, steps*sizeof(*buffer));
                    memcpy(t, chan->t, steps*sizeof(*t));
                    memcpy(&dchan, chan->chan, sizeof(dchan));
                    break;
                }
            }
            rt_make_hard_real_time();
            run->dirty = 0;
        }

        // Wait for sync
        rt_sem_signal(run->presync);
        if ( !chan || !steps ) {
            continue;
        }
        rt_sem_wait(run->sync);

        expected = rt_get_time() + t[step = 0];
        while ( run->running ) {
            ret = comedi_data_write(run->dev, dchan.subdevice, dchan.channel,
                                    dchan.range, dchan.aref, buffer[step]);
            if ( !ret ) { // Fatal: Write failure
                run->running = 0;
                break;
            }

            // Wait period
            now = rt_get_time();
            if ( now < expected ) {
                rt_sleep(expected-now);
            }

            if ( ++step == steps ) { // Return to base value before leaving
                ret = comedi_data_write(run->dev, dchan.subdevice, dchan.channel,
                                        dchan.range, dchan.aref, buffer[0]);
                run->running = 0;
                break;
            }

            expected += t[step];
        }
    }

    rt_make_soft_real_time();
    rt_thread_delete(task);

    if ( ! ret ) {
        perror("Error writing to AO, thread exited");
        return (void *)EXIT_FAILURE;
    }

    return 0;
}
