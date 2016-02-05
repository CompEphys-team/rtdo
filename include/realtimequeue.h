/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-02-04

--------------------------------------------------------------------------*/
#ifndef REALTIMEQUEUE_H
#define REALTIMEQUEUE_H

#include <memory>

#ifdef CONFIG_RT
#include <rtai_mbx.h>
#else
#include <queue>
#endif

template <typename T> class RealtimeQueue
{
public:
    /** SizeAdjustPolicy defines how the RT mailbox size is adjusted if its size limit is reached:
     * Append: Increase the size by 1
     * Double: Double the size
     * Tenfold: Increase the size by an order of magnitude
     * NoAdjust: Never increase the size
     **/
    enum SizeAdjustPolicy { Append, Double, Tenfold, NoAdjust };

    //!< Initialise a queue (RT: mailbox) for @a size instances of T. In non-RT builds, the arguments do nothing.
    RealtimeQueue(int size = 1000, SizeAdjustPolicy policy = Double);
    ~RealtimeQueue();

    //!< Copy a queue, maintaining a common internal queue
    RealtimeQueue(const RealtimeQueue<T> &);

    /** Send @a message to the queue. In non-RT builds, additional arguments are ignored.
     * If @a resize is true (default) and the queue is full, the queue will be resized according to the policy set
     * at construction. The call never blocks, but may take a while to complete, as resizing causes reallocation.
     * If @a resize is false and @a blocking is true, the call will wait until the message can be sent.
     * If @a resize is false, @a blocking is false, and the queue is full, the call will return immediately, discarding
     * the message and logging the incident. Appropriate resizing will occur on the next call to @fn push or @fn flush
     * that allows it.
     * @return true if the message was sent successfully.
     **/
    bool push(T message, bool resize = true, bool blocking = false);

    /** Attempt to send @a message, waiting at most @a nanos nanoseconds before returning.
     * No resizing occurs on this call. Redirects to @fn push on non-RT builds.
     * @return true if the message was sent successfully.
     **/
    bool push_timed(T message, long nanos);

    /** Attempt to receive @a message from the queue. If @a blocking is true (default), the call will wait until the
     * message is received. Otherwise, the call will return immediately.
     * No resizing occurs on this call.
     * @return true if the message was received successfully.
     **/
    bool pop(T &message, bool blocking = true);

    /** Attempt to receive @a message, waiting at most @a nanos nanoseconds before returning.
     * No resizing occurs on this call. Redirects to @fn pop on non-RT builds.
     * @return true if the message was received successfully.
     **/
    bool pop_timed(T &message, long nanos);

    /** Flush the queue, removing all messages from it. If @a resize is true (default), deferred resizing from @fn push
     * is performed.
     **/
    void flush(bool resize = true);

private:

#ifdef CONFIG_RT
    std::shared_ptr<MBX> mbx;
    std::shared_ptr<int> size;
    std::shared_ptr<int> overruns;
    const RealtimeQueue<T>::SizeAdjustPolicy pol;

    void resize();
#else
    std::shared_ptr<std::queue<T>> q;
#endif
};


// ---------------------- Implementations below, because templates -----------------------------------------------------
#ifdef CONFIG_RT

template <typename T> RealtimeQueue<T>::RealtimeQueue(int size, SizeAdjustPolicy p) :
    mbx(rt_typed_mbx_init(0, size * sizeof(T), PRIO_Q), &rt_mbx_delete),
    size(new int(size)),
    overruns(new int(0)),
    pol(p)
{}

template <typename T> RealtimeQueue<T>::RealtimeQueue(const RealtimeQueue<T> &other) :
    mbx(other.mbx),
    size(other.size),
    overruns(other.overruns),
    pol(other.pol)
{}

template <typename T> RealtimeQueue<T>::~RealtimeQueue() {}

template <typename T> void RealtimeQueue<T>::resize()
{
    if ( !*overruns )
        return;

    int newsize = *size;
    while ( newsize < *size + *overruns ) {
        switch ( pol ) {
        case RealtimeQueue::Append:
            newsize++;
            break;
        case RealtimeQueue::Double:
            newsize *= 2;
            break;
        case RealtimeQueue::Tenfold:
            newsize *= 10;
            break;
        default:
            return;
        }
    }
    *overruns = 0;

    MBX *newbox = rt_typed_mbx_init(0, newsize * sizeof(T), PRIO_Q);
    char buffer[*size * sizeof(T)];
    int bytesNotReceived = rt_mbx_receive_wp(&*mbx, buffer, *size * sizeof(T));
    rt_mbx_send(newbox, buffer, *size * sizeof(T) - bytesNotReceived);
    mbx.reset(newbox, &rt_mbx_delete);
    *size = newsize;
}

template <typename T> bool RealtimeQueue<T>::push(T message, bool doResize, bool blocking)
{
    if ( doResize && pol != NoAdjust ) {
        if ( rt_mbx_send_if(&*mbx, &message, sizeof(T)) ) {
            (*overruns)++;
            resize();
            return push(message, true);
        } else {
            return true;
        }
    } else {
        if ( blocking ) {
            return !rt_mbx_send(&*mbx, &message, sizeof(T));
        } else {
            if ( rt_mbx_send_if(&*mbx, &message, sizeof(T)) ) {
                (*overruns)++;
                return false;
            } else {
                return true;
            }
        }
    }
}

template <typename T> bool RealtimeQueue<T>::push_timed(T message, long nanos)
{
    return !rt_mbx_send_timed(&*mbx, &message, sizeof(T), nano2count(nanos));
}

template <typename T> bool RealtimeQueue<T>::pop(T &message, bool blocking)
{
    if ( blocking ) {
        return !rt_mbx_receive(&*mbx, &message, sizeof(T));
    } else {
        return !rt_mbx_receive_if(&*mbx, &message, sizeof(T));
    }
}

template <typename T> bool RealtimeQueue<T>::pop_timed(T &message, long nanos)
{
    return !rt_mbx_receive_timed(&*mbx, &message, sizeof(T), nano2count(nanos));
}

template <typename T> void RealtimeQueue<T>::flush(bool doResize)
{
    char buffer[*size * sizeof(T)];
    rt_mbx_receive_wp(&*mbx, buffer, *size * sizeof(T));
    if ( doResize )
        resize();
}

#else

template <typename T> class RealtimeQueue<T>::Impl
{
public:
    std::shared_ptr<std::queue<T>> q;

    Impl(const Impl<T> &other) :
        q(other.q)
    {}
};

template <typename T> RealtimeQueue<T>::RealtimeQueue(int, SizeAdjustPolicy):
    q(new std::queue<T>)
{}

template <typename T> RealtimeQueue<T>::~RealtimeQueue() {}

template <typename T> RealtimeQueue<T>::RealtimeQueue(const RealtimeQueue<T> &other) :
    q(other.q)
{}

template <typename T> bool RealtimeQueue<T>::push(T message, bool, bool)
{
    q->push(message);
    return true;
}

template <typename T> bool RealtimeQueue<T>::push_timed(T message, long) { return push(message); }

template <typename T> bool RealtimeQueue<T>::pop(T &message)
{
    message = q->front();
    q->pop();
    return true;
}

template <typename T> bool RealtimeQueue<T>::pop_timed(T &message, long)
{
    return pop(message);
}

template <typename T> void RealtimeQueue<T>::flush(bool)
{
    for ( std::queue<T>::size_type i = 0, k = q->size(); i < k; ++i )
        q->pop();
}

#endif

#endif // REALTIMEQUEUE_H
