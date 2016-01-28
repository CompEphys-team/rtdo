/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-25

--------------------------------------------------------------------------*/
#include "channel.h"
#include "config.h"
#include "realtimeenvironment.h"

#ifdef CONFIG_RT
// ----------- Realtime implementation --------------
#include "impl/channel_impl.h"

#else
// ----------------------- Non-realtime implementation ----------------------

class Channel::Impl
{
public:
    Impl(Channel::Type type, int, unsigned int, unsigned int, Channel::Aref) {
        if ( type != Channel::Simulator )
            throw RealtimeException(RealtimeException::RuntimeMsg, "Non-RT channels can only be simulators.");
    }
};

bool Channel::setDirection(Channel::Type)
{
    // NYI
    return false;
}

void Channel::flush()
{
    // NYI
}

double Channel::nextSample()
{
    // NYI
    return 0.0;
}

bool Channel::setDevice(int deviceno) { return true; }
bool Channel::setChannel(unsigned int channel) { return true; }
bool Channel::setRange(unsigned int range) { return true; }
bool Channel::setAref(Channel::Aref aref) { return true; }
void Channel::setConversionFactor(double) {}
void Channel::setOffset(double) {}
void Channel::setOffsetSource(int) {}

bool Channel::hasDirection(Channel::Type type) const { return false; }
unsigned int Channel::numChannels() const { return 0; }
bool Channel::hasRange(unsigned int range) const { return false; }
bool Channel::hasAref(Channel::Aref aref) const { return false; }

double Channel::rangeMin(unsigned int range) const { return 0; }
double Channel::rangeMax(unsigned int range) const { return 0; }
std::string Channel::rangeUnit(unsigned int range) const { return std::string(); }

double Channel::offset() const { return 0; }
double Channel::conversionFactor() const { return 0; }
int Channel::device() const { return 0; }
unsigned int Channel::channel() const { return 0; }
unsigned int Channel::range() const { return 0; }
Channel::Aref Channel::aref() const { return Ground; }
int Channel::offsetSource() const { return 0; }

lsampl_t Channel::convert(double voltage) const { return 0; }
double Channel::convert(lsampl_t sample) const { return 0; }

void Channel::readOffset() {}

bool Channel::read(lsampl_t &sample, bool hint) const { return false; }
bool Channel::write(lsampl_t sample) const { return false; }
void Channel::put(lsampl_t &sample) {}

#endif

int Channel::nextID = 1;

Channel::Channel(Type type, int deviceno, unsigned int channel, unsigned int range, Aref aref) :
    _type(type),
    _waveform(new inputSpec()),
    _waveformChanged(new bool(true)),
    _ID(nextID++),
    pImpl(new Impl(type, deviceno, channel, range, aref))
{}

Channel::Channel(int ID, Type type, int deviceno, unsigned int channel, unsigned int range, Aref aref) :
    Channel(type, deviceno, channel, range, aref)
{
    _ID = ID;
    if ( nextID <= ID )
        nextID = ID + 1;
}

Channel::Channel(const Channel &other) :
    _type(other._type),
    _waveform(other._waveform),
    _waveformChanged(other._waveformChanged),
    _name(other._name),
    _ID(other._ID),
    pImpl(new Impl(*other.pImpl))
{}

Channel &Channel::operator=(const Channel &other)
{
    _type = other._type;
    _waveform = other._waveform;
    _waveformChanged = other._waveformChanged;
    _name = other._name;
    _ID = other._ID;
    pImpl.reset(new Impl(*other.pImpl));
    return *this;
}

Channel::~Channel() {}
