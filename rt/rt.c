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

#include "rt_helper.h"
#include "converter.h"
#include "rtdo_types.h"

#include "HHVClampParameters.h"

#define DO_RT_CPUS 0xF000
#define DO_MAIN_PRIO 5
#define DO_AO_PRIO 0
#define DO_AI_PRIO 0
#define DO_MAX_CHANNELS 32
#define DO_MIN_AI_BUFSZ 100

//-------------------------------------
// Comedi globals
#define DO_SAMP_NS ((RTIME)(DT * 1e6))

comedi_t *dev;
char *device;

//------------------------------------
// Thread globals
RT_TASK *maintask;

int end = 0;

SEM *sync_sem;

int max_ai_bufsz;
static int ai_supersampling = 1;

//------------------------------------
// Channel specification

static rtdo_channel *channels[DO_MAX_CHANNELS];
static int num_channels = 1;

static rtdo_thread_runinfo ai_runinfo;
static rtdo_thread_runinfo ao_runinfo;

//------------------------------------
// Functions
void cleanup(int init_level);
void *ao_fun(void *);
void *ai_fun(void *);

void rtdo_init(const char *device_file, const char *calibration_file) {
    int ret = 0, init_level = 0;

    // Load converter
    init_level++;
    ret = rtdo_converter_init(calibration_file);
    if ( ret == DOE_LOAD_CALIBRATION ) {
        printf("Warning: Calibration could not be read. Conversion may be imprecise.\n");
    } else if ( ret == DOE_LOAD_LIBRARY || ret == DOE_LOAD_FUNC ) {
        printf("Failed to load comedilib module. Is libcomedi installed in an accessible path?\n");
        exit(EXIT_FAILURE);
    }

    // Open in real time
    init_level++;
    if ( ! (dev = comedi_open(device_file)) ) {
        printf("Failed to open comedi device %s.\n", device_file);
        cleanup(init_level);
        exit(EXIT_FAILURE);
    }
    device = strdup(device_file);

    // Set up RT
    init_level++;
    rt_allow_nonroot_hrt();
    maintask = rt_task_init(nam2num("MAIN"), DO_MAIN_PRIO, 5000, 0);
    if (!maintask) {
        printf("Main RT setup failed. Is the rtai_sched module installed?\n");
        return cleanup(init_level);
    }
    rt_set_oneshot_mode();
    start_rt_timer(0);
    signal(SIGINT | SIGKILL, rtdo_stop);

    // Test semaphore function
    init_level++;
    sync_sem = rt_typed_sem_init(nam2num("SYNC"), 0, BIN_SEM);
    rt_sem_signal(sync_sem);
    if ( rt_sem_wait(sync_sem) != 1 ) {
        printf("Semaphore setup failed. Is the rtai_sem module installed?\n");
        cleanup(init_level);
        exit(EXIT_FAILURE);
    }

    // Test mailbox function
    MBX *mbx = rt_typed_mbx_init(nam2num("TESTBX"), sizeof(lsampl_t), FIFO_Q);
    lsampl_t msg = 1;
    rt_mbx_send(mbx, &msg, sizeof(lsampl_t));
    msg = 0;
    rt_mbx_receive(mbx, &msg, sizeof(lsampl_t));
    rt_mbx_delete(mbx);
    if ( !msg ) {
        printf("Mailbox setup failed. Is the rtai_mbx module installed?\n");
        cleanup(init_level);
        exit(EXIT_FAILURE);
    }

    init_level++;
    ai_runinfo.presync = rt_typed_sem_init(nam2num("AIPRE"), 0, BIN_SEM);
    ai_runinfo.load = rt_typed_sem_init(nam2num("AILOAD"), 0, BIN_SEM);
    ai_runinfo.running = 0;
    ai_runinfo.dirty = 1;
    ai_runinfo.thread = rt_thread_create(ai_fun, NULL, 1000);

    ao_runinfo.presync = rt_typed_sem_init(nam2num("AOPRE"), 0, BIN_SEM);
    ao_runinfo.load = rt_typed_sem_init(nam2num("AOLOAD"), 0, BIN_SEM);
    ao_runinfo.running = 0;
    ai_runinfo.dirty = 1;
    ao_runinfo.thread = rt_thread_create(ao_fun, NULL, 1000);

    if ( !ai_runinfo.thread || !ao_runinfo.thread ) {
        printf("RT thread setup failed.\n");
        cleanup(init_level);
        exit(EXIT_FAILURE);
    }

    rt_make_soft_real_time();
}

int rtdo_create_channel(enum rtdo_channel_type type,
                        unsigned int subdevice_offset,
                        unsigned int channel,
                        unsigned int range,
                        unsigned int reference,
                        double conversion_factor,
                        double offset,
                        int buffer_size) {
    enum comedi_subdevice_type subdev_type;
    int subdev, ret;
    rtdo_channel *chan;

    if ( num_channels == DO_MAX_CHANNELS ) {
        printf("Error: Too many channels allocated.\n");
        return 0;
    }

    // Find subdevice
    if ( type == DO_CHANNEL_AI ) {
        subdev_type = COMEDI_SUBD_AI;
    } else if ( type == DO_CHANNEL_AO ) {
        subdev_type = COMEDI_SUBD_AO;
    }
    subdev = comedi_find_subdevice_by_type(dev, subdev_type, subdevice_offset);
    if (subdev < 0) {
        printf("Subdevice not found. Is the rtai_comedi module installed?\n");
        return 0;
    }

    if ( (chan = malloc(sizeof(rtdo_channel))) == NULL )
        return 0;

    if ( type == DO_CHANNEL_AO ) {
        if ( buffer_size < 1 )
            buffer_size = 1;
        if ( (chan->t = malloc(buffer_size * sizeof(RTIME))) == NULL )
            return 0;
        if ( (chan->buffer = malloc(buffer_size * sizeof(lsampl_t))) == NULL )
            return 0;
        chan->numsteps = 0;
        chan->mbx = 0;
    } else if ( type == DO_CHANNEL_AI ) {
        if ( buffer_size < DO_MIN_AI_BUFSZ )
            buffer_size = DO_MIN_AI_BUFSZ;
        chan->t = 0;
        chan->buffer = 0;
        chan->numsteps = 0;
        chan->mbx = rt_typed_mbx_init(0, buffer_size * sizeof(lsampl_t), PRIO_Q);
        if ( buffer_size > max_ai_bufsz )
            max_ai_bufsz = buffer_size;
    }

    chan->type = type;
    chan->subdevice = subdev;
    chan->channel = channel;
    chan->range = range;
    chan->aref = reference;
    chan->active = 0;
    chan->bufsz = buffer_size;

    // Load converter
    ret = rtdo_converter_create(device, chan, conversion_factor, offset);
    if ( ret == DOE_OPEN_DEV ) {
        printf("Failed to open comedi device in non-realtime mode.\n");
        return 0;
    } else if ( ret == DOE_MEMORY ) {
        return 0;
    }
    
    int idx = num_channels++;
    channels[idx] = chan;
    rtdo_set_channel_active(idx, 1);

    return idx;
}

int rtdo_set_channel_active(int handle, int active) {
    if ( handle < 1 || handle >= num_channels || !channels[handle] ) {
        printf("Error: Invalid channel handle.\n");
        return -1;
    }
    int ret = channels[handle]->active;
    if ( active >= 0 ) {
        channels[handle]->active = !!active;
        if ( channels[handle]->active != ret ) {
            if ( channels[handle]->type == DO_CHANNEL_AI )
                ai_runinfo.dirty = 1;
            else if ( channels[handle]->type == DO_CHANNEL_AO )
                ao_runinfo.dirty = 1;
        }
    }
    return ret;
}

void rtdo_set_supersampling(int multiplier) {
    if ( multiplier < 1 )
        multiplier = 1;
    ai_supersampling = multiplier;
    ai_runinfo.dirty = 1;
}

void rtdo_stop(int unused) {
    cleanup(0);
}

void cleanup( int init_level ) {
    int i;
    switch(init_level) {
        default:
        case 0:
        case 5:
            end = 1;
            ai_runinfo.running = 0;
            ao_runinfo.running = 0;

            if ( ai_runinfo.load )    rt_sem_delete(ai_runinfo.load);
            if ( ao_runinfo.load )    rt_sem_delete(ao_runinfo.load);
            if ( sync_sem )           rt_sem_delete(sync_sem);
            if ( ai_runinfo.presync ) rt_sem_delete(ai_runinfo.presync);
            if ( ao_runinfo.presync ) rt_sem_delete(ao_runinfo.presync);

            if ( ai_runinfo.thread)   rt_thread_join(ai_runinfo.thread);
            if ( ao_runinfo.thread)   rt_thread_join(ao_runinfo.thread);

            for ( i = 1; i < num_channels; i++ ) {
                if ( channels[i] ) {
                    if ( channels[i]->mbx )
                        rt_mbx_delete(channels[i]->mbx);
                    free(channels[i]->converter);
                    free(channels[i]->buffer);
                    free(channels[i]->t);
                    free(channels[i]);
                }
            }

        case 4:
            if ( init_level == 4 )
                rt_sem_delete(sync_sem);

        case 3:
            rt_thread_delete(maintask);
            stop_rt_timer();

        case 2:
            comedi_close(dev);

        case 1:
            rtdo_converter_exit();
    }
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

float rtdo_channel_get(int handle) {
    if ( handle < 1 || handle >= num_channels || !channels[handle] ) {
        printf("Error: Invalid channel handle.\n");
        return 0.0;
    }
    if ( channels[handle]->type != DO_CHANNEL_AI ) {
        printf("Error: Data acquisition is only supported for AI channels.\n");
        return 0.0;
    }

    lsampl_t sample;
    rt_mbx_receive(channels[handle]->mbx, &sample, sizeof(lsampl_t));
    return (float)rtdo_convert_to_physical(sample, channels[handle]->converter);
}

int rtdo_set_stimulus(int handle, double baseVal, int numsteps, double *values, double *ms, double ms_total) {
    if ( handle < 1 || handle >= num_channels || !channels[handle] ) {
        printf("Error: Invalid channel handle.\n");
        return EINVAL;
    }
    if ( channels[handle]->type != DO_CHANNEL_AO ) {
        printf("Error: Stimulus output is only supported for AO channels.\n");
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

    chan->buffer[0] = rtdo_convert_from_physical(baseVal, chan->converter);
    for ( i = 0; i < numsteps; i++ ) {
        chan->buffer[i+1] = rtdo_convert_from_physical(values[i], chan->converter);
        chan->t[i] = nano2count((RTIME)(ms[i]*1e6)) - t;
        t += chan->t[i];
    }
    chan->t[numsteps] = nano2count((RTIME)(ms_total*1e6)) - t;
    chan->numsteps = numsteps+1;

    return 0;
}

void *ao_fun(void *unused) {
    RT_TASK *task;
    int ret = 1, i, step, steps=0, bufsz=0;
    rtdo_channel *chan;
    RTIME now, expected, *t=0;
    lsampl_t *buffer=0;

    task = rt_thread_init(nam2num("AOFUN"), DO_AO_PRIO, 5000, SCHED_FIFO, DO_RT_CPUS);
    if (!task) {
        printf("AO RT setup failed.\n");
        return (void *)EXIT_FAILURE;
    }

    mlockall(MCL_CURRENT | MCL_FUTURE);
    rt_make_hard_real_time();

    while ( !end ) {
        // Load new channel config
        if ( ao_runinfo.dirty ) {
            rt_sem_wait(ao_runinfo.load);
            rt_make_soft_real_time();
            chan = 0;
            for ( i = 1; i < num_channels; i++ ) {
                if ( channels[i]->type == DO_CHANNEL_AO && channels[i]->active ) {
                    chan = channels[i];
                    if ( chan->numsteps > bufsz ) {
                        free(buffer);
                        free(t);
                        bufsz = chan->numsteps;
                        if ( !(buffer = malloc(bufsz*sizeof(*buffer)))
                             || !(t = malloc(bufsz*sizeof(*t))) ) {
                            rt_make_soft_real_time();
                            rt_thread_delete(task);
                            printf("Error: AO out of memory.\n");
                            return (void *)ENOMEM;
                        }
                    }
                    steps = chan->numsteps;
                    memcpy(buffer, chan->buffer, steps*sizeof(*buffer));
                    memcpy(t, chan->t, steps*sizeof(*t));
                    break;
                }
            }
            rt_make_hard_real_time();
            ao_runinfo.dirty = 0;
        }

        // Wait for sync
        rt_sem_signal(ao_runinfo.presync);
        if ( !chan || !steps ) {
            continue;
        }
        rt_sem_wait(sync_sem);

        expected = rt_get_time() + t[step = 0];
        while ( ao_runinfo.running ) {
            ret = comedi_data_write(dev, chan->subdevice, chan->channel, chan->range, chan->aref, buffer[step]);
            if ( !ret ) { // Fatal: Write failure
                end = 0;
                ao_runinfo.running = 0;
                break;
            }

            // Wait period
            now = rt_get_time();
            if ( now < expected ) {
                rt_sleep(expected-now);
            }

            if ( ++step == steps ) { // Return to base value before leaving
                ret = comedi_data_write(dev, chan->subdevice, chan->channel, chan->range, chan->aref, buffer[0]);
                if ( ! ret ) {
                    end = 0;
                }
                ao_runinfo.running = 0;
                break;
            }

            expected += t[step];
        }
    }

    rt_make_soft_real_time();
    rt_thread_delete(task);

    if ( ! ret ) {
        printf("Error writing to AO, thread exited.\n");
        return (void *)EXIT_FAILURE;
    }

    return 0;

}

void *ai_fun(void *unused) {
    RT_TASK *task;
    int ret = 1, i, nchans, iter;
    rtdo_channel *chans[DO_MAX_CHANNELS];
    RTIME now, expected, samp_ticks = nano2count(DO_SAMP_NS);
    lsampl_t sample, sums[DO_MAX_CHANNELS];

    task = rt_thread_init(nam2num("AIFUN"), DO_AI_PRIO, 5000, SCHED_FIFO, DO_RT_CPUS);
    if (!task) {
        printf("AI RT setup failed.\n");
        return (void *)EXIT_FAILURE;
    }

    mlockall(MCL_CURRENT | MCL_FUTURE);
    rt_make_hard_real_time();

    while ( !end ) {
        // Load new channel config
        if ( ai_runinfo.dirty ) {
            rt_sem_wait(ai_runinfo.load);
            nchans = 0;
            for ( i = 1; i < num_channels; i++ ) {
                if ( channels[i]->type == DO_CHANNEL_AI && channels[i]->active ) {
                    chans[nchans++] = channels[i];
                }
            }
            samp_ticks = nano2count(DO_SAMP_NS / ai_supersampling);
            ai_runinfo.dirty = 0;
        }

        iter = 0;
        for ( i = 0; i < nchans; i++ )
            sums[i] = 0;

        // Wait for sync
        rt_sem_signal(ai_runinfo.presync);
        rt_sem_wait(sync_sem);

        expected = rt_get_time() + samp_ticks;
        while ( ai_runinfo.running ) {
            // Read samples
            for ( i = 0; i < nchans; i++ ) {
                if ( nchans > 1 ) {
                    comedi_data_read_hint(dev, chans[i]->subdevice, chans[i]->channel,
                                          chans[i]->range, chans[i]->aref);
                }
                ret = comedi_data_read(dev, chans[i]->subdevice, chans[i]->channel, chans[i]->range,
                                       chans[i]->aref, &sample);
                if ( ! ret ) { // Fatal: Read failed.
                    end = 0;
                    ai_runinfo.running = 0;
                    break;
                }
                if ( ai_supersampling > 1 ) {
                    sums[i] += sample;
                    if ( (iter+1) % ai_supersampling == 0 ) {
                        sample = sums[i] / ai_supersampling;
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

    if ( ! ret ) {
        printf("Error reading from AI, thread exited.\n");
        return (void *)EXIT_FAILURE;
    }

    return 0;
}
