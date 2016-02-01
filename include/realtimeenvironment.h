/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-25

--------------------------------------------------------------------------*/
#ifndef REALTIMEENVIRONMENT_H
#define REALTIMEENVIRONMENT_H

#include <memory>
#include <exception>
#include "channel.h"

//!< Note: RealtimeEnvironment is a singleton class. Access is provided through RealtimeEnvironment::env().
class RealtimeEnvironment
{
public:
    inline static RealtimeEnvironment &env()
    {
        static RealtimeEnvironment instance;
        return instance;
    }

    ~RealtimeEnvironment();
    RealtimeEnvironment(RealtimeEnvironment const&) = delete;
    void operator=(RealtimeEnvironment const&) = delete;

    void sync();
    void pause();
    void setSupersamplingRate(int acquisitionsPerSample);

    //!< References to the input and output channel in use
    Channel &outChannel() const;
    Channel &inChannel(int index = 0) const;

    /** Use @a c starting from next call to @fn RealtimeEnvironment::sync.
     * Note: At any given time, only one output channel can be active. Previously set output channels are therefore discarded.
     **/
    void addChannel(Channel &c);
    void clearChannels();

    struct comedi_t_struct *getDevice(int deviceno, bool RT); //!< Opens and returns a comedi device in hard/soft realtime [RT]
    std::string getDeviceName(int deviceno);
    unsigned int getSubdevice(int deviceno, Channel::Type type); //!< Finds the first subdevice of the given type [RT]

private:
    RealtimeEnvironment();
    class Impl;
    std::unique_ptr<Impl> pImpl;
};


class RealtimeException : std::exception
{
public:
    enum type { Setup, MailboxTest, SemaphoreTest, RuntimeFunc, RuntimeMsg, Timeout } errtype;
    std::string funcname;
    int errval;

    RealtimeException(type t, std::string funcname = "", int errval = 0);
    const char *what() const noexcept;
};

#endif // REALTIMEENVIRONMENT_H
