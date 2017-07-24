#include "queue.h"
#include <queue>
#include <atomic>
#include <new>
#include <time.h>

namespace RTMaybe
{

template <typename T>
class Queue<T>::Impl
{
public:
    Impl(size_t size) : SIZE(size), _data(new T[size]), _read(0), _write(0)
    {
    }

    ~Impl()
    {
        delete _data;
    }

    bool push(T message)
    {
        while ( !push_if(message) );
        return true;
    }

    bool push_if(T message)
    {
        const auto current_write = _write.load(std::memory_order_relaxed);
        const auto next_write = increment(current_write);
        if(next_write != _read.load(std::memory_order_acquire)) {
            _data[current_write] = message;
            _write.store(next_write, std::memory_order_release);
            return true;
        }
        return false; // full queue
    }

    bool push_timed(T message, long nanos)
    {
        timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        long target = ts.tv_nsec + nanos;
        bool ret = push_if(message);
        while ( !ret && ts.tv_nsec < target )
            ret = push_if(message);
        return ret;
    }

    bool pop(T& message)
    {
        while ( !pop_if(message) );
        return true;
    }

    bool pop_if(T& message)
    {
        const auto current_read = _read.load(std::memory_order_relaxed);
        if(current_read == _write.load(std::memory_order_acquire))
            return false; // empty queue
        message = _data[current_read];
        _read.store(increment(current_read), std::memory_order_release);
        return true;
    }

    bool pop_timed(T& message, long nanos)
    {
        timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        long target = ts.tv_nsec + nanos;
        bool ret = pop_if(message);
        while ( !ret && ts.tv_nsec < target )
            ret = pop_if(message);
        return ret;
    }

    void flush()
    {
        _read.store(0);
        _write.store(0);
    }

    void resize(int newSize)
    {
        if ( SIZE < newSize ) {
            delete _data;
            _data = new T[newSize];
            SIZE = newSize;
        }
    }

private:
    inline size_t increment(size_t i) const
    {
        return (i + 1) % SIZE;
    }

    size_t SIZE;
    T *_data;
    std::atomic<size_t> _read;
    std::atomic<size_t> _write;
};



template <typename T>
Queue<T>::Queue(int sz, const char*) : mbx(new Impl(sz)) {}

template <typename T>
Queue<T>::~Queue()
{
    delete mbx;
}

template <typename T>
bool Queue<T>::push(T message)
{
    return mbx->push(message);
}

template <typename T>
bool Queue<T>::push_if(T message)
{
    return mbx->push_if(message);
}

template <typename T>
bool Queue<T>::push_timed(T message, long nanos)
{
    return mbx->push_timed(message, nanos);
}

template <typename T>
bool Queue<T>::pop(T &message)
{
    return mbx->pop(message);
}

template <typename T>
bool Queue<T>::pop_if(T &message)
{
    return mbx->pop_if(message);
}

template <typename T>
bool Queue<T>::pop_timed(T &message, long nanos)
{
    return mbx->pop_timed(message, nanos);
}

template <typename T>
void Queue<T>::flush(bool)
{
    return mbx->flush();
}

template <typename T>
void Queue<T>::resize(int newSize)
{
    return mbx->resize(newSize);
}

}

#include <comedi.h>
#include "types.h"
template class RTMaybe::Queue<lsampl_t>;
template class RTMaybe::Queue<DataPoint>;
