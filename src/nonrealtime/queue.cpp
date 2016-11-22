#include "queue.h"
#include <queue>

namespace RTMaybe
{

template <typename T>
class Queue<T>::Impl : public std::queue<T> {};

template <typename T>
Queue<T>::Queue(int, const char*) {}

template <typename T>
Queue<T>::~Queue()
{
    sendsem.broadcast();
    rcvsem.broadcast();
    sendsem.wait();
    rcvsem.wait();
}

template <typename T>
bool Queue<T>::push(T message)
{
    sendsem.wait();
    mbx->push(message);
    sendsem.signal();
    return true;
}

template <typename T>
bool Queue<T>::push_if(T message)
{
    if ( !sendsem.wait_if() )
        return false;
    mbx->push(message);
    sendsem.signal();
    return true;
}

template <typename T>
bool Queue<T>::pop(T &message)
{
    rcvsem.wait();
    message = mbx->front();
    mbx->pop();
    rcvsem.signal();
    return true;
}

template <typename T>
bool Queue<T>::pop_if(T &message)
{
    if ( !rcvsem.wait_if() )
        return false;
    message = mbx->front();
    mbx->pop();
    rcvsem.signal();
    return true;
}

template <typename T>
void Queue<T>::flush(bool)
{
    sendsem.wait();
    rcvsem.wait();
    while ( !mbx->empty() )
        mbx->pop();
    sendsem.signal();
    rcvsem.signal();
}

template <typename T>
void Queue<T>::resize(int) {}

}
