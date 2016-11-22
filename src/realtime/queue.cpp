#include "queue.h"

extern "C" {
#include <rtai_mbx.h>
}

namespace RTMaybe
{

template <typename T>
class Queue<T>::Impl : public MBX {};

template <typename T>
Queue<T>::Queue(int size, const char *name) :
    nam(name ? nam2num(name) : 0),
    mbx((Impl*)rt_typed_mbx_init(nam, size * sizeof(T), PRIO_Q)),
    size(size),
    overruns(0)
{
    if ( !mbx )
        throw std::runtime_error("RTAI mailbox setup failed. Is the rtai_mbx module active?");
}

template <typename T>
Queue<T>::~Queue()
{
    rcvsem.broadcast();
    sendsem.broadcast();
    rcvsem.wait();
    sendsem.wait();
    rt_mbx_delete(mbx);
}

template <typename T>
bool Queue<T>::push(T message)
{
    sendsem.wait();
    bool fail = rt_mbx_send(mbx, &message, sizeof(T));
    sendsem.signal();
    return !fail;
}

template <typename T>
bool Queue<T>::push_if(T message)
{
    if ( !sendsem.wait_if() )
        return false;
    bool fail = rt_mbx_send_if(mbx, &message, sizeof(T));
    if ( fail )
        ++overruns;
    sendsem.signal();
    return !fail;
}

template <typename T>
bool Queue<T>::push_timed(T message, long nanos)
{
    if ( !sendsem.wait_timed(nanos) )
        return false;
    bool fail = rt_mbx_send_timed(mbx, &message, sizeof(T), nano2count(nanos));
    if ( fail )
        ++overruns;
    sendsem.signal();
    return !fail;
}

template <typename T>
bool Queue<T>::pop(T &message)
{
    rcvsem.wait();
    bool fail = rt_mbx_receive(mbx, &message, sizeof(T));
    rcvsem.signal();
    return !fail;
}

template <typename T>
bool Queue<T>::pop_if(T &message)
{
    if ( !rcvsem.wait_if() )
        return false;
    bool fail = rt_mbx_receive_if(mbx, &message, sizeof(T));
    rcvsem.signal();
    return !fail;
}

template <typename T>
bool Queue<T>::pop_timed(T &message, long nanos)
{
    if ( !rcvsem.wait_timed(nanos) )
        return false;
    bool fail = rt_mbx_receive_timed(mbx, &message, sizeof(T), nano2count(nanos));
    rcvsem.signal();
    return !fail;
}

template <typename T>
void Queue<T>::flush(bool doResize)
{
    sendsem.wait();
    rcvsem.wait();
    char buffer[size * sizeof(T)];
    rt_mbx_receive_wp(mbx, buffer, size * sizeof(T));
    if ( doResize && overruns )
        mResize(size + overruns);
    overruns = 0;
    sendsem.signal();
    rcvsem.signal();
}

template <typename T>
void Queue<T>::resize(int newsize)
{
    sendsem.wait();
    rcvsem.wait();
    mResize(newsize);
    sendsem.signal();
    rcvsem.signal();
}

template <typename T>
void Queue<T>::mResize(int newsize)
{
    char buffer[size * sizeof(T)];
    int bytesNotReceived = rt_mbx_receive_wp(mbx, buffer, size * sizeof(T));
    rt_mbx_delete(mbx);

    MBX *newbox = rt_typed_mbx_init(nam, newsize * sizeof(T), PRIO_Q);
    rt_mbx_send(newbox, buffer, size * sizeof(T) - bytesNotReceived);
    mbx = (Impl*)newbox;
    size = newsize;
}

}

#include <comedi.h>
template class RTMaybe::Queue<lsampl_t>;
