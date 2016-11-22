#ifndef QUEUE_H
#define QUEUE_H

#include <memory>
#include "conditionvariable.h"

// Forward
struct rt_mailbox;

namespace RTMaybe {
    template <typename T> class Queue;
}

template <typename T> class RTMaybe::Queue
{
public:

    //!< Initialise a queue (RT: mailbox) for @a size instances of T.
    Queue(int size = 1000, const char *name = 0);
    ~Queue();

    /** Send @a message to the queue, waiting until the message can be sent.
     * @return true if the message was sent successfully.
     **/
    bool push(T message);

    /** Send @a message to the queue, returning immediately.
     * [RT only] If the queue is full, the message is discarded, and Queue::overruns increased.
     * @return true if the message was sent successfully.
     **/
    bool push_if(T message);

    /** Attempt to send @a message, waiting at most @a nanos nanoseconds before returning.
     * Note that if a higher-priority task is receiving data from the queue, the delay may be up to twice @a nanos.
     * @return true if the message was sent successfully.
     **/
    bool push_timed(T message, long nanos);

    /** Attempt to receive @a message from the queue, waiting until the message is received.
     * @return true if the message was received successfully.
     **/
    bool pop(T &message);

    /** Attempt to receive @a message from the queue, returning immediately.
     * @return true if the message was received successfully.
     **/
    bool pop_if(T &message);

    /** Attempt to receive @a message, waiting at most @a nanos nanoseconds before returning.
     * Note that if a higher-priority task is sending data to the queue, the delay may be up to twice @a nanos.
     * @return true if the message was received successfully.
     **/
    bool pop_timed(T &message, long nanos);

    /** Flush the queue, removing all messages from it. If @a resize is true (default), deferred resizing from @fn push
     * is performed.
     **/
    void flush(bool resize = true);

    /** Resize the queue, allocating space for a minimum of @a size instances of T.
     * If the queue is already larger than @a size, does nothing.
     */
    void resize(int newsize);

    inline int overrun() { return overruns; }

private:
    class Impl;
    unsigned long nam;
    Impl *mbx;
    int size;
    int overruns;
    RTMaybe::ConditionVariable rcvsem, sendsem;



    void mResize(int newsize);
};

#endif // QUEUE_H
