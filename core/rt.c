/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2015-11-17

--------------------------------------------------------------------------*/

#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <rtai_comedi.h>
#include <rtai_sem.h>
#include <rtai_mbx.h>

#include "rt.h"
#include "types.h"
#include "softrtdaq.h"

static comedi_t *dev=0;
static char *devfile = "/dev/comedi0";

static RT_TASK *maintask;
static SEM *sync_sem;

static int max_ai_bufsz;

static rtdo_channel *channels[DO_MAX_CHANNELS];
static int num_channels = 1;

static rtdo_thread_runinfo ai_runinfo;
static rtdo_thread_runinfo ao_runinfo;

void rtdo_sigint(int unused);
void cleanup(int init_level);

void *ao_fun(void *);
void *ai_fun(void *);

int rtdo_init(const char *device) {
    int init_level = 0;

    if ( dev ) {
        cleanup(0);
    }
    
    if ( !device )
        device = devfile;
    else
        devfile = strdup(device);

    // Open in real time
    init_level++;
    if ( ! (dev = comedi_open(device)) ) {
        perror("Failed to open comedi device in RT mode");
        return ENODEV;
    }

    // Set up RT
    init_level++;
    rt_allow_nonroot_hrt();
    maintask = rt_task_init(nam2num("MAIN"), DO_MAIN_PRIO, 5000, 0);
    if (!maintask) {
        perror("Main RT setup failed. Is the rtai_sched module installed?");
        cleanup(init_level);
        return EXIT_FAILURE;
    }
    rt_set_oneshot_mode();
    start_rt_timer(0);
    signal(SIGINT | SIGKILL, cleanup);

    // Test semaphore function
    init_level++;
    sync_sem = rt_typed_sem_init(nam2num("SYNC"), 0, BIN_SEM);
    rt_sem_signal(sync_sem);
    if ( rt_sem_wait(sync_sem) != 1 ) {
        perror("Semaphore setup failed. Is the rtai_sem module installed?");
        cleanup(init_level);
        return EXIT_FAILURE;
    }

    // Test mailbox function
    MBX *mbx = rt_typed_mbx_init(nam2num("TESTBX"), sizeof(lsampl_t), FIFO_Q);
    lsampl_t msg = 1;
    rt_mbx_send(mbx, &msg, sizeof(lsampl_t));
    msg = 0;
    rt_mbx_receive(mbx, &msg, sizeof(lsampl_t));
    rt_mbx_delete(mbx);
    if ( !msg ) {
        perror("Mailbox setup failed. Is the rtai_mbx module installed?");
        cleanup(init_level);
        return EXIT_FAILURE;
    }

    init_level++;
    ai_runinfo.presync = rt_typed_sem_init(nam2num("AIPRE"), 0, BIN_SEM);
    ai_runinfo.load = rt_typed_sem_init(nam2num("AILOAD"), 0, BIN_SEM);
    ai_runinfo.sync = sync_sem;
    ai_runinfo.running = 0;
    ai_runinfo.dirty = 1;
    ai_runinfo.exit = 0;
    ai_runinfo.chans = channels;
    ai_runinfo.num_chans =& num_channels;
    ai_runinfo.dev = dev;
    ai_runinfo.samp_ticks = nano2count(DO_SAMP_NS_DEFAULT);
    ai_runinfo.supersampling = 1;
    ai_runinfo.thread = rt_thread_create(ai_fun, &ai_runinfo, 1000);

    ao_runinfo.presync = rt_typed_sem_init(nam2num("AOPRE"), 0, BIN_SEM);
    ao_runinfo.load = rt_typed_sem_init(nam2num("AOLOAD"), 0, BIN_SEM);
    ao_runinfo.sync = sync_sem;
    ao_runinfo.running = 0;
    ao_runinfo.dirty = 1;
    ao_runinfo.exit = 0;
    ao_runinfo.chans = channels;
    ao_runinfo.num_chans =& num_channels;
    ao_runinfo.dev = dev;
    ao_runinfo.thread = rt_thread_create(ao_fun, &ao_runinfo, 1000);

    if ( !ai_runinfo.thread || !ao_runinfo.thread ) {
        perror("RT thread setup failed");
        cleanup(init_level);
        return EXIT_FAILURE;
    }

    rt_make_soft_real_time();

    return 0;
}

void rtdo_exit() {
    cleanup(0);
}

void cleanup( int init_level ) {
    if ( ! dev )
        return;
    int i;
    switch(init_level) {
    default:
    case 0:
    case 4:
        rt_make_hard_real_time();

        ai_runinfo.exit = 1;
        ao_runinfo.exit = 1;
        ai_runinfo.running = 0;
        ao_runinfo.running = 0;

        if ( ai_runinfo.load )    rt_sem_delete(ai_runinfo.load);
        if ( ao_runinfo.load )    rt_sem_delete(ao_runinfo.load);
        if ( sync_sem )           rt_sem_delete(sync_sem);
        if ( ai_runinfo.presync ) rt_sem_delete(ai_runinfo.presync);
        if ( ao_runinfo.presync ) rt_sem_delete(ao_runinfo.presync);

        if ( ai_runinfo.thread )  rt_thread_join(ai_runinfo.thread);
        if ( ao_runinfo.thread )  rt_thread_join(ao_runinfo.thread);

        for ( i = 1; i < num_channels; i++ ) {
            if ( channels[i] ) {
                if ( channels[i]->mbx )
                    rt_mbx_delete(channels[i]->mbx);
                free(channels[i]->buffer);
                free(channels[i]->t);
                free(channels[i]);
            }
        }
        num_channels = 1;

    case 3:
        if ( sync_sem )
            rt_sem_delete(sync_sem);

    case 2:
        rt_make_soft_real_time();
        rt_thread_delete(maintask);
        stop_rt_timer();

    case 1:
        comedi_close(dev);
        dev = 0;
    }
}

int rtdo_add_channel(daq_channel *dchan, int buffer_size) {
    rtdo_channel *chan;
    int ret, idx;
    
    if ( !dev && !(ret = rtdo_init(NULL)) )
        return ret;

    if ( num_channels == DO_MAX_CHANNELS ) {
        perror("Too many channels allocated");
        return EMLINK;
    }

    if ( !(chan = malloc(sizeof(rtdo_channel))) ) {
        return ENOMEM;
    }

    if ( dchan->type == COMEDI_SUBD_AO ) {
        if ( buffer_size < 1 )
            buffer_size = 1;
        if ( !(chan->t = malloc(buffer_size * sizeof(RTIME))) ) {
            return ENOMEM;
        }
        if ( !(chan->buffer = malloc(buffer_size * sizeof(lsampl_t))) ) {
            return ENOMEM;
        }
        chan->numsteps = 0;
        chan->mbx = 0;
    } else if ( dchan->type == COMEDI_SUBD_AI ) {
        if ( buffer_size < DO_MIN_AI_BUFSZ )
            buffer_size = DO_MIN_AI_BUFSZ;
        chan->t = 0;
        chan->buffer = 0;
        chan->numsteps = 0;
        chan->mbx = rt_typed_mbx_init(0, buffer_size * sizeof(lsampl_t), PRIO_Q);
        if ( buffer_size > max_ai_bufsz )
            max_ai_bufsz = buffer_size;
    }

    chan->active = 0;
    chan->bufsz = buffer_size;
    chan->chan = dchan;
    
    idx = num_channels++;
    channels[idx] = chan;
    dchan->handle = idx;
    rtdo_set_channel_active(idx, 1);
    return 0;
}

int rtdo_set_channel_active(int handle, int active) {
    if ( handle < 1 || handle >= num_channels || !channels[handle] ) {
        perror("Invalid channel handle");
        return -EINVAL;
    }
    int ret = channels[handle]->active;
    if ( active >= 0 ) {
        channels[handle]->active = !!active;
        if ( channels[handle]->active != ret ) {
            if ( channels[handle]->chan->type == COMEDI_SUBD_AI )
                ai_runinfo.dirty = 1;
            else if ( channels[handle]->chan->type == COMEDI_SUBD_AO )
                ao_runinfo.dirty = 1;
        }
    }
    return ret;
}

void rtdo_set_supersampling(int multiplier) {
    if ( multiplier < 1 )
        multiplier = 1;
    ai_runinfo.supersampling = multiplier;
    ai_runinfo.dirty = 1;
}
void rtdo_set_sampling_rate(double ms_per_sample, int acquisitions_per_sample) {
    if (acquisitions_per_sample < 1)
        acquisitions_per_sample = 1;
    RTIME ticks = nano2count((RTIME)(ms_per_sample * 1e6));
    ai_runinfo.samp_ticks = ticks;
    ai_runinfo.supersampling = acquisitions_per_sample;
    ai_runinfo.dirty = 1;
}

void rtdo_stop() {
    ai_runinfo.running = 0;
    ao_runinfo.running = 0;
}

void rtdo_sync() {
    rt_make_hard_real_time();

    // Stop threads
    ao_runinfo.running = 0;
    ai_runinfo.running = 0;

    // Load new assignments, if any
    if ( ao_runinfo.dirty )
        rt_sem_signal(ao_runinfo.load);
    if ( ai_runinfo.dirty )
        rt_sem_signal(ai_runinfo.load);

    // Flush the message buffer
    int i;
    lsampl_t buf[max_ai_bufsz];
    rt_sem_wait(ai_runinfo.presync);
    for ( i = 1; i < num_channels; i++ )
        if ( channels[i] && channels[i]->active && channels[i]->mbx )
            rt_mbx_receive_wp(channels[i]->mbx, &buf, channels[i]->bufsz * sizeof(lsampl_t));

    // Wait for AO to load
    rt_sem_wait(ao_runinfo.presync);

    // Release the hounds!
    ao_runinfo.running = 1;
    ai_runinfo.running = 1;
    rt_sem_broadcast(sync_sem);

    rt_make_soft_real_time();
}

float rtdo_get_data(int handle, int *err) {
    if ( handle < 1 || handle >= num_channels || !channels[handle] ) {
        perror("Invalid channel handle");
        *err = EINVAL;
        return 0.0;
    }
    if ( channels[handle]->chan->type != COMEDI_SUBD_AI ) {
        perror("Data acquisition is only supported for AI channels");
        *err = EPERM;
        return 0.0;
    }

    lsampl_t sample;
    RTIME delay = 100 * ai_runinfo.samp_ticks;
    if ( rt_mbx_receive_timed(channels[handle]->mbx, &sample, sizeof(lsampl_t), delay) ) {
        perror("Read from queue timed out");
        *err = EBUSY;
        return 0.0;
    }
    return (float)daq_convert_to_physical(sample, channels[handle]->chan);
}

float rtdo_read_now(int handle, int *err) {
    if ( handle < 1 || handle >= num_channels || !channels[handle] ) {
        perror("Invalid channel handle");
        *err = EINVAL;
        return 0.0;
    }
    if ( channels[handle]->chan->type != COMEDI_SUBD_AI ) {
        perror("Data acquisition is only supported for AI channels");
        *err = EPERM;
        return 0.0;
    }
    if ( ai_runinfo.running ) {
        perror("Realtime data acquisition is running");
        *err = EBUSY;
        return 0.0;
    }

    lsampl_t sample;
    comedi_data_read_hint(dev, channels[handle]->chan->subdevice, channels[handle]->chan->channel,
                          channels[handle]->chan->range, channels[handle]->chan->aref);
    comedi_data_read(dev, channels[handle]->chan->subdevice, channels[handle]->chan->channel,
                     channels[handle]->chan->range, channels[handle]->chan->aref, &sample);
    return (float)daq_convert_to_physical(sample, channels[handle]->chan);
}

int rtdo_set_stimulus(int handle, double baseVal, int numsteps, double *values, double *ms, double ms_total) {
    if ( handle < 1 || handle >= num_channels || !channels[handle] ) {
        perror("Invalid channel handle");
        return EINVAL;
    }
    if ( channels[handle]->chan->type != COMEDI_SUBD_AO ) {
        perror("Stimulus output is only supported for AO channels");
        return EINVAL;
    }

    rtdo_channel *chan = channels[handle];
    RTIME t = 0;
    int i;

    if ( numsteps >= chan->bufsz ) {
        free(chan->buffer);
        free(chan->t);
        if ( !(chan->buffer = malloc((numsteps+1)*sizeof(*chan->buffer))) )
            return ENOMEM;
        if ( !(chan->t = malloc((numsteps+1)*sizeof(*chan->t))) )
            return ENOMEM;
    }

    int err=0;
    chan->buffer[0] = daq_convert_from_physical(baseVal, chan->chan, &err);
    if ( err )
        return err;
    for ( i = 0; i < numsteps; i++ ) {
        chan->buffer[i+1] = daq_convert_from_physical(values[i], chan->chan, &err);
        if ( err )
            return err;
        chan->t[i] = nano2count((RTIME)(ms[i]*1e6)) - t;
        t += chan->t[i];
    }
    chan->t[numsteps] = nano2count((RTIME)(ms_total*1e6)) - t;
    chan->numsteps = numsteps+1;

    return 0;
}
