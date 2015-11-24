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

#include "rt_helper.h"
#include "converter.h"
#include "options.h"

#include "HHVClampParameters.h"

#define DO_AI_CPUS 0xF000
#define DO_AO_CPUS 0xF000
#define DO_GPU_CPUS 0x0FF0

//-------------------------------------
// Comedi globals
#define DO_BUFSZ 10000
#define DO_SAMP_NS (DT * 1e-6)

comedi_t *dev;
int aodev;

static rtdo_converter_type *outchan_vclamp_Vc_converter;
static rtdo_converter_type *inchan_vclamp_Im_converter;

lsampl_t ai_buffer[DO_BUFSZ];
float Vset; // Command voltage in mV

//------------------------------------
// Thread globals
RT_TASK *maintask;
pthread_t ai_thread=0;
int end=0;

SEM *ai_sem;

MBX *ai_mbx;
int ai_msg_size;

//------------------------------------
// Functions
void cleanup(int init_level);
void *ai_fun(void *init_level);

void rtdo_init(void) {
    int ret = 0, init_level = 0;

    // Set up comedi
    printf("Setting up comedi...\n");

    // Load converters
    init_level++;
    ret = rtdo_converter_init(daqopts.calibration_file);
    if ( ret == DOE_LOAD_CALIBRATION ) {
        printf("Warning: Calibration could not be read. Conversion may be imprecise.\n");
    } else if ( ret == DOE_LOAD_LIBRARY || ret == DOE_LOAD_FUNC ) {
        printf("Failed to load comedilib module. Is libcomedi installed in an accessible path?\n");
        exit(EXIT_FAILURE);
    }
    inchan_vclamp_Im_converter = rtdo_converter_create(daqopts.device, &inchan_vclamp_Im, &ret);
    if ( ret == DOE_OPEN_DEV ) {
        printf("Failed to open comedi device in non-realtime mode. Is the driver running?\n");
        cleanup(init_level);
        exit(EXIT_FAILURE);
    } else if ( ret == DOE_FIND_SUBDEV ) {
        printf("Failed to find AI subdevice at offset %d.\n", inchan_vclamp_Im.subdevice_offset);
        cleanup(init_level);
        exit(EXIT_FAILURE);
    }
    outchan_vclamp_Vc_converter = rtdo_converter_create(daqopts.device, &outchan_vclamp_Vc, &ret);
    if ( ret == DOE_OPEN_DEV ) {
        printf("Failed to open comedi device in non-realtime mode. Is the driver running?\n");
        cleanup(init_level);
        exit(EXIT_FAILURE);
    } else if ( ret == DOE_FIND_SUBDEV ) {
        printf("Failed to find AO subdevice at offset %d.\n", outchan_vclamp_Vc.subdevice_offset);
        cleanup(init_level);
        exit(EXIT_FAILURE);
    }

    // Open in real time
    init_level++;
    if ( ! (dev = comedi_open(daqopts.device)) ) {
        printf("Failed to open comedi device %s.\n", daqopts.device);
        cleanup(init_level);
        exit(EXIT_FAILURE);
    }
    aodev = comedi_find_subdevice_by_type(dev, COMEDI_SUBD_AO, outchan_vclamp_Vc.subdevice_offset);
    rtdo_set_Vout(0.0);
    printf("Done.\n");

    // Set up RT
    printf("Setting up main task RT...\n");
    init_level++;
    rt_allow_nonroot_hrt();
    maintask = rt_task_init(nam2num("MAIN"), 0, 5000, 0);
    if (!maintask) {
        printf("Main RT setup failed. Is the rtai_sched module installed?\n");
        return cleanup(init_level);
    }
    rt_set_oneshot_mode();
    start_rt_timer(0);
    signal(SIGINT | SIGKILL, rtdo_stop);
    printf("Done.\n");

    // Set up semaphores
    init_level++;
    ai_sem = rt_typed_sem_init(nam2num("AISEM"), 0, BIN_SEM);
    rt_sem_signal(ai_sem);
    if ( ! rt_sem_wait(ai_sem) ) {
        printf("Semaphore setup failed. Is the rtai_sem module installed?\n");
        cleanup(init_level);
        exit(EXIT_FAILURE);
    }

    // Set up mailboxes
    init_level++;
    ai_msg_size = sizeof(lsampl_t);
    ai_mbx = rt_typed_mbx_init(nam2num("AIMBX"), ai_msg_size, FIFO_Q);
    lsampl_t msg = 1;
    rt_mbx_send(ai_mbx, &msg, ai_msg_size);
    msg = 0;
    rt_mbx_receive(ai_mbx, &msg, ai_msg_size);
    if ( !msg ) {
        printf("Mailbox setup failed. Is the rtai_mbx module installed?\n");
        cleanup(init_level);
        exit(EXIT_FAILURE);
    }

    // Start subordinates
    printf("Launching AI thread.\n");
    ai_thread = rt_thread_create(ai_fun, (void *)(unsigned long)(init_level++), 1000);
}

void rtdo_stop(int unused) {
    rtdo_set_Vout(0.0);
    cleanup(0);
}

void cleanup( int init_level ) {
    switch(init_level) {
        default:
        case 0:
        case 6:
            end = 1;
            printf("Waiting for threads to exit, please wait...\n");
            if (ai_thread) rt_thread_join(ai_thread);
            printf("All threads complete.\n");

        case 5:
            rt_mbx_delete(ai_mbx);
            printf("Mailboxes deleted.\n");

        case 4:
            rt_sem_delete(ai_sem);
            printf("Semaphores released.\n");

        case 3:
            rt_thread_delete(maintask);
            stop_rt_timer();
            printf("RT stopped.\n");

        case 2:
            comedi_close(dev);

        case 1:
            rtdo_converter_exit();
            printf("Comedi closed.\n");
    }
}

void rtdo_sync() {
    static running = 0;
    if ( ! running ) {
        rt_sem_signal(ai_sem);
        running = 1;
    } else {
        lsampl_t buf[100];
        while ( rt_mbx_receive_wp(ai_mbx, &buf, 100*ai_msg_size) > 0 ) {
            // Flush buffer.
        }
    }
}

float rtdo_get_Im(float t) {
    lsampl_t sample;
    rt_mbx_receive(ai_mbx, &sample, ai_msg_size);
    return (float)rtdo_convert_to_physical(sample, inchan_vclamp_Im_converter);
}

void rtdo_set_Vout(float V) {
    if ( V == Vset )
        return;
    comedi_data_write(dev, aodev,
                      outchan_vclamp_Vc.channel, outchan_vclamp_Vc.range, outchan_vclamp_Vc.reference,
                      rtdo_convert_from_physical((double) V, outchan_vclamp_Vc_converter) );
    Vset = V;
}

void *ai_fun(void *init_level) {
    RT_TASK *task;
    int aidev, ret, i = 0;
    RTIME now, expected;

    // Open subdevice
    aidev = comedi_find_subdevice_by_type(dev, COMEDI_SUBD_AI, inchan_vclamp_Im.subdevice_offset);
    if (aidev < 0) {
        printf("Analog in subdevice not found. Is the rtai_comedi module installed?\n");
        cleanup((unsigned long) init_level);
        exit(EXIT_FAILURE);
    }

    // Enter realtime
    printf("AI entering realtime!\n");
    task = rt_thread_init(nam2num("AIFUN"), 0, 5000, SCHED_FIFO, DO_AI_CPUS);
    if (!task) {
        printf("AI RT setup failed.\n");
        cleanup((unsigned long) init_level);
        exit(EXIT_FAILURE);
    }
    mlockall(MCL_CURRENT | MCL_FUTURE);
    rt_make_hard_real_time();

    // Wait for sync
    rt_sem_wait(ai_sem);

    expected = rt_get_time_ns() + DO_SAMP_NS;
    while ( !end ) {
        // Read sample to buffer
        ret = comedi_data_read(dev, aidev, inchan_vclamp_Im.channel, inchan_vclamp_Im.range,
                               inchan_vclamp_Im.reference, &(ai_buffer[i]));
        if ( ! ret ) {
            rt_printk("Error reading from AI, exiting.\n");
            break;
        }

        rt_mbx_send(ai_mbx, &(ai_buffer[i]), ai_msg_size);

        // Move to next entry
        if ( i == DO_BUFSZ )
            i = 0;

        // Wait period
        now = rt_get_time_ns();
        if ( now < expected ) {
            rt_sleep(expected-now);
        }
        expected += DO_SAMP_NS;
    }

    end = 1;
    rt_make_soft_real_time();
    rt_thread_delete(task);

    printf("AI exiting.\n");
    return 0;
}
