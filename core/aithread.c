/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2015-12-03

--------------------------------------------------------------------------*/
#include "RC_rtai_comedi.h"
#include <rtai_sem.h>
#include <rtai_mbx.h>
#include "rt.h"
#include "softrtdaq.h"

void *ai_fun(void *runinfo) {
    RT_TASK *task;
    int ret = 1, i, nchans=0, iter, nsum=0;
    rtdo_channel *chans[DO_MAX_CHANNELS];
    daq_channel dchans[DO_MAX_CHANNELS];
    RTIME now, expected, samp_ticks=0;
    lsampl_t sample, sums[DO_MAX_CHANNELS];
    rtdo_thread_runinfo *run = (rtdo_thread_runinfo *)runinfo;

    for ( i = 0; i < DO_MAX_CHANNELS; i++ ) {
        daq_create_channel(&dchans[i]);
    }

    task = rt_thread_init(nam2num("AIFUN"), DO_AI_PRIO, 5000, SCHED_FIFO, DO_RT_CPUS);
    if (!task) {
        perror("AI RT setup failed");
        return (void *)EXIT_FAILURE;
    }

    mlockall(MCL_CURRENT | MCL_FUTURE);
    rt_make_hard_real_time();

    while ( !run->exit ) {
        // Load new channel config
        if ( run->dirty ) {
            rt_sem_wait(run->load);
            nchans = 0;
            for ( i = 1; i < *(run->num_chans); i++ ) {
                if ( run->chans[i]->chan->type == COMEDI_SUBD_AI && run->chans[i]->active ) {
                    chans[nchans] = run->chans[i];
                    daq_copy_channel(&dchans[nchans], run->chans[i]->chan);
                    nchans++;
                }
            }
            nsum = run->supersampling;
            samp_ticks = (RTIME)(run->samp_ticks / nsum);
            run->dirty = 0;
        }

        iter = 0;
        for ( i = 0; i < nchans; i++ )
            sums[i] = 0;

        // Wait for sync
        rt_sem_signal(run->presync);
        rt_sem_wait(run->sync);

        expected = rt_get_time() + samp_ticks;
        while ( run->running ) {
            // Read samples
            for ( i = 0; i < nchans; i++ ) {
                if ( nchans > 1 ) {
                    RC_comedi_data_read_hint(chans[i]->dev, dchans[i].subdevice, dchans[i].channel,
                                          dchans[i].range, dchans[i].aref);
                }
                ret = RC_comedi_data_read(chans[i]->dev, dchans[i].subdevice, dchans[i].channel,
                                       dchans[i].range, dchans[i].aref, &sample);
                if ( ! ret ) { // Fatal: Read failed.
                    run->running = 0;
                    break;
                }
                if ( nsum > 1 ) {
                    sums[i] += sample;
                    if ( (iter+1) % nsum == 0 ) {
                        sample = sums[i] / nsum;
                        sums[i] = 0;
                        rt_mbx_send(chans[i]->mbx, &sample, sizeof(lsampl_t));
                    }
                } else {
                    rt_mbx_send(chans[i]->mbx, &sample, sizeof(lsampl_t));
                }
            }

            // Wait period
            iter++;
            now = rt_get_time();
            if ( now < expected ) {
                rt_sleep(expected-now);
            }
            expected += samp_ticks;
        }
    }

    rt_make_soft_real_time();
    rt_thread_delete(task);

    for ( i = 0; i < DO_MAX_CHANNELS; i++ ) {
        daq_delete_channel(&dchans[i]);
    }

    if ( ! ret ) {
        perror("Error reading from AI, thread exited");
        return (void *)EXIT_FAILURE;
    }

    return 0;
}
