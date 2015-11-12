#include <rtai_comedi.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>

#include "gpgpu.h"

#define DYNO_AI_CPUS 0xF000
#define DYNO_AO_CPUS 0xF000
#define DYNO_GPU_CPUS 0x0FF0

/**
 * Comedi globals
 */
#define DYNO_DEV "/dev/comedi0"
#define DYNO_AOCHAN 0
#define DYNO_AORANGE 0
#define DYNO_AICHAN 0
#define DYNO_AIRANGE 0

#define DYNO_N_SAMP 2000
#define DYNO_SAMP_NS 50000
#define DYNO_BUFFER_SIZE 2

comedi_t *dev;

lsampl_t ai_buffer[DYNO_BUFFER_SIZE*DYNO_N_SAMP];
lsampl_t ao_buffer[DYNO_BUFFER_SIZE*DYNO_N_SAMP];

lsampl_t ao_maxdata;

/**
 * Thread globals
 */
RT_TASK *maintask;
pthread_t ao_thread=0;
pthread_t ai_thread=0;
pthread_t gpu_thread=0;
int end=0;

RTIME timeout = 3; // Thread timeout in segment durations

SEM *ai_sem, *ao_sem;
short ai_segment_written = 0;
short ao_segment_written = 0;

/**
 * Functions
 */
void *ao_fun(void *);
void *ai_fun(void *);
void *gpu_fun(void *);
int cleanup(int);

void endme(int foo) {
    end = 1;
    cleanup(0);
}

int main(void) {
    timeout = nano2count(timeout * DYNO_N_SAMP * DYNO_SAMP_NS);

    // Set up comedi
    printf("Setting up comedi...\n");
    if ( ! (dev = comedi_open(DYNO_DEV)) ) {
        printf("Failed to open comedi device %s.\n", DYNO_DEV);
        return cleanup(1);
    }
    printf("Done.\n");

    // Initialise cuda
    printf("Setting up cuda...\n");
    gpu_init(DYNO_N_SAMP);
    printf("Done.\n");

    // Set up main task as soft real time thread
    printf("Setting up main task RT...\n");
    rt_allow_nonroot_hrt();
    maintask = rt_task_init(nam2num("MAIN"), 0, 5000, 0);
    if (!maintask) {
        printf("Main RT setup failed. Is the rtai_sched module installed?\n");
        return cleanup(3);
    }
    rt_set_oneshot_mode();
    start_rt_timer(0);
    signal(SIGINT | SIGKILL, endme);
    printf("Done.\n");

    // Set up inter-thread communication channels
    ao_sem = rt_typed_sem_init(nam2num("AOSEM"), 0, BIN_SEM);
    ai_sem = rt_typed_sem_init(nam2num("AISEM"), 0, BIN_SEM);
    rt_sem_signal(ai_sem);
    if ( ! rt_sem_wait(ai_sem) ) {
        printf("Semaphore setup failed. Is the rtai_sem module installed?\n");
        return cleanup(4);
    }

    // Start subordinates
    printf("Launching threads.\n");
    ao_thread = rt_thread_create(ao_fun, NULL, 1000);
    ai_thread = rt_thread_create(ai_fun, NULL, 1000);
    gpu_thread = rt_thread_create(gpu_fun, NULL, 100000);

    // Get user input to cancel
    printf("Press return to halt execution.\n");
    getchar();

    return cleanup(0);
}

int cleanup( int stage ) {
    end = 1;

    switch(stage) {
        default:
        case 0:
        case 5:
            stage = 0;
            printf("Waiting for threads to exit, please wait...\n");
            if (ao_thread) rt_thread_join(ao_thread);
            if (ai_thread) rt_thread_join(ai_thread);
            if (gpu_thread) rt_thread_join(gpu_thread);
            printf("All threads complete.\n");

        case 4:
            rt_sem_delete(ao_sem);
            rt_sem_delete(ai_sem);
            printf("Semaphores released.\n");

        case 3:
            stop_rt_timer();
            rt_thread_delete(maintask);
            printf("RT stopped.\n");

        case 2:
            gpu_exit();
            printf("Cuda closed.\n");

        case 1:
            comedi_close(dev);
            printf("Comedi closed.\n");
    }

    return stage;
}

void *ai_fun(void *arg) {
    RT_TASK *task;
    int aidev, ret;
    lsampl_t ai_maxdata;
    long long i = 0;
    short k = 0;
    RTIME now, expected;

    // Open subdevice
    aidev = comedi_find_subdevice_by_type(dev, COMEDI_SUBD_AI, 0);
    if (aidev < 0) {
        printf("Analog in subdevice not found. Is the rtai_comedi module installed?\n");
        exit(1);
    }
    ai_maxdata = comedi_get_maxdata(dev, aidev, DYNO_AICHAN);

    // Enter realtime
    printf("AI entering realtime!\n");
    task = rt_thread_init(nam2num("AIFUN"), 0, 5000, SCHED_FIFO, DYNO_AI_CPUS);
    if (!task) {
        printf("Analog in RT setup failed.\n");
        exit(1);
    }
    mlockall(MCL_CURRENT | MCL_FUTURE);
    rt_make_hard_real_time();

    expected = rt_get_time_ns() + DYNO_SAMP_NS;
    while ( !end ) {
        // Read sample to buffer
        ret = comedi_data_read(dev, aidev, DYNO_AICHAN, DYNO_AIRANGE, AREF_DIFF, &(ai_buffer[i]));
        if ( ! ret ) {
            rt_printk("Error reading from AI, exiting.\n");
            break;
        }

        // Move to next entry
        i++;
        if ( !(i % DYNO_N_SAMP) ) {
            // Send "segment k ready"
            ai_segment_written = k;
            rt_sem_signal(ai_sem);
            k++;
            if ( k == DYNO_BUFFER_SIZE ) {
                k = 0;
                i = 0;
            }
        }

        // Wait period
        now = rt_get_time_ns();
        if ( now < expected ) {
            //rt_busy_sleep(expected-now);
            rt_sleep(expected-now);
        }
        expected += DYNO_SAMP_NS;
    }

    end = 1;
    rt_make_soft_real_time();
    rt_thread_delete(task);

    printf("AI exiting.\n");
    return 0;
}

void *ao_fun(void *arg) {
    RT_TASK *task;
    RTIME now, expected;
    int aodev;
    short k;
    int i = 0;

    // Open subdevice
    aodev = comedi_find_subdevice_by_type(dev, COMEDI_SUBD_AO, 0);
    if (aodev < 0) {
        printf("Analog out subdevice not found. Is the rtai_comedi module installed?\n");
        exit(1);
    }
    ao_maxdata = comedi_get_maxdata(dev, aodev, DYNO_AOCHAN);

    // Enter realtime
    printf("AO entering realtime!\n");
    task = rt_thread_init(nam2num("AOFUN"), 0, 5000, SCHED_FIFO, DYNO_AO_CPUS);
    if (!task) {
        printf("Analog out RT setup failed.\n");
        exit(1);
    }
    mlockall(MCL_CURRENT | MCL_FUTURE);
    rt_make_hard_real_time();

    rt_sem_wait_timed(ao_sem, timeout);
    k = ao_segment_written;
    i = DYNO_N_SAMP * k;
    while ( !end ) {
        // Write sample to ao
        comedi_data_write(dev, aodev, DYNO_AOCHAN, DYNO_AORANGE, AREF_GROUND, ao_buffer[i]);
        i++;

        if ( !(i % DYNO_N_SAMP) ) {
            // Wait for gpu_thread's output data buffer
            rt_sem_wait_timed(ao_sem, timeout);
            k = ao_segment_written;
            i = DYNO_N_SAMP * k;
            expected = rt_get_time_ns() + DYNO_SAMP_NS;
        } else {
            // Wait period
            now = rt_get_time_ns();
            if ( now < expected ) {
                //rt_busy_sleep(expected-now);
                rt_sleep(expected-now);
            }
            expected += DYNO_SAMP_NS;
        }
    }

    end = 1;
    rt_make_soft_real_time();
    rt_thread_delete(task);

    printf("AO exiting.\n");
    return 0;
}

void *gpu_fun(void *arg) {
    RT_TASK *task;
    short k;

    printf("GPU entering realtime!\n");
    task = rt_thread_init(nam2num("GPUFUN"), 0, 5000, SCHED_FIFO, DYNO_GPU_CPUS);
    if (!task) {
        printf("Cuda soft RT setup failed.\n");
        exit(1);
    }
    mlockall(MCL_CURRENT | MCL_FUTURE);
    rt_make_hard_real_time();

    rt_sem_wait_timed(ai_sem, timeout);
    k = ai_segment_written;
    while( !end ) {
        rt_make_soft_real_time();
        gpu_process(&(ai_buffer[k * DYNO_N_SAMP]), &(ao_buffer[k * DYNO_N_SAMP]), ao_maxdata);
        rt_make_hard_real_time();

        // confirm ao buffer written
        ao_segment_written = k;
        rt_sem_signal(ao_sem);

        // Wait for buffer segment readiness
        rt_sem_wait_timed(ai_sem, timeout);
        k = ai_segment_written;
    }

    end = 1;
    rt_make_soft_real_time();
    rt_thread_delete(task);

    printf("GPU exiting.\n");
    return 0;
}
