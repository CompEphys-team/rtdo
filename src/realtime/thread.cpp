#include "thread.h"
#include "conditionvariable.h"
#include <QString>
#include <iostream>

extern "C" {
#include <rtai_lxrt.h>
}

namespace RTMaybe
{

class Thread::Impl
{
public:
    Impl(void *(*fn)(void *), void *arg, ThreadData p) :
        joinable(false),
        fn(fn),
        arg(arg),
        p(p),
        cv(new RTMaybe::ConditionVariable)
    {
        id = rt_thread_create((void *)&Impl::launchStatic, this, p.stackSize);
        if ( !id )
            throw std::runtime_error("RTAI thread setup failed. Is the rtai_sched kernel module active?");

        cv->wait();
        if ( !joinable ) // Launch failed
            throw std::runtime_error("RTAI thread launch failed. Is the rtai_sched kernel module active?");

        delete cv;
        cv = 0;
    }

    long id;
    bool joinable;

private:
    void *(*fn)(void *);
    void *arg;

    ThreadData p;

    RTMaybe::ConditionVariable *cv;

    static void *launchStatic(Thread::Impl *_this)
    {
        return _this->launch();
    }

    void *launch()
    {
        RT_TASK *task;
        task = rt_thread_init(p.name, p.priority, p.maxMessageSize, p.policy, p.cpusAllowed);
        if ( !task ) {
            cv->signal();
            // Defer throwing to the constructor
            return (void *)EXIT_FAILURE;
        }

        joinable = true;
        rt_make_hard_real_time();
        cv->signal();

        void *ret;
        try {
            ret = fn(arg);
        } catch (std::runtime_error e) {
            char name[10] = "";
            if ( p.name ) {
                name[0] = ' ';
                num2nam(p.name, &name[1]);
            }
            std::cerr << "An uncaught exception occurred in child thread " << name << ": " << e.what() << std::endl;
            rt_thread_delete(task);
            exit(EINTR);
        }

        rt_thread_delete(task);
        return ret;
    }
};

Thread::Thread(void *(*fn)(void *), void *arg, ThreadData p) :
    pImpl(new Impl(fn, arg, p))
{}

Thread::~Thread()
{
    if ( joinable() )
        join();
}

void *Thread::join()
{
    if ( !pImpl->joinable )
        return (void *)ESRCH;
    // Since rt_thread_join is just a wrapper that throws away the return value of pthread_join:
    void *tret;
    int jret = pthread_join(pImpl->id, &tret);
    pImpl->joinable = false;
    if ( jret )
        throw std::runtime_error(QString("Failed to join thread: pthread_join returned ").arg(jret).toStdString());
    return tret;
}

bool Thread::joinable() const
{
    return pImpl->joinable;
}

void Thread::initEnv(ThreadData p)
{
    static std::unique_ptr<RT_TASK, int(*)(RT_TASK*)> task(nullptr, &rt_thread_delete);

    if ( !task.get() ) {
        rt_allow_nonroot_hrt();
        task.reset(rt_task_init_schmod(nam2num("RTDO"), p.priority, p.stackSize, p.maxMessageSize, p.policy, p.cpusAllowed));
        if ( !task.get() )
            throw std::runtime_error("RTAI thread setup failed. Is the rtai_sched kernel module active?");
        rt_set_oneshot_mode();
        if ( !rt_is_hard_timer_running() )
            start_rt_timer(0);
    } else {
        task.release();
        task.reset(rt_task_init_schmod(nam2num("RTDO"), p.priority, p.stackSize, p.maxMessageSize, p.policy, p.cpusAllowed));
        if ( !task.get() )
            throw std::runtime_error("RTAI thread reinit failed.");
    }
    rt_make_soft_real_time();
}

}
