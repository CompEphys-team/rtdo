/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-25

--------------------------------------------------------------------------*/
#ifndef CHANNEL_H
#define CHANNEL_H

#include <memory>
#include "shared.h"

#ifdef CONFIG_RT
#include <comedi.h>
#else
typedef unsigned int lsampl_t;
#endif

//!< Note: In non-RT build, Channel is an empty husk of nothingness.
class Channel
{
public:
    enum Direction { AnalogIn, AnalogOut };
    enum Aref { Ground, Common, Diff, Other };

    Channel(Direction direction, int deviceno = 0, unsigned int channel = 0, unsigned int range = 0, Aref aref = Ground);
    Channel(int ID, Direction direction, int deviceno, unsigned int channel, unsigned int range, Aref aref);

    //!< Note: Copies of a channel maintain a reference to a common sample (read) queueand a common waveform.
    Channel(const Channel&);
    Channel &operator=(const Channel&);

    ~Channel();

    /** All setX return true if the requested value could be set given higher-order values. Dependency is as follows:
     *     Device depends on the existence of a device
     *     Direction depends on Device support
     *     Channel depends on Device and Direction
     *     Aref depends on Device and Direction
     *     Range depends on Device, Direction and Channel
     * Dependent values (e.g. Range upon calling setChannel) are reset if necessary, such that Channel is always a valid object.
     */
    bool setDirection(Direction direction);
    bool setDevice(int deviceno);
    bool setChannel(unsigned int channel);
    bool setRange(unsigned int range);
    bool setAref(Aref aref);

    //!< Dependency-free setters
    void setOffsetSource(int ID); //!< Use channel @a ID to set the offset upon calling Channel::readOffset, or 0 to disable
    void setOffset(double offset);
    void setConversionFactor(double factor);
    inline void setWaveform(const inputSpec &i) { *_waveform = i; *_waveformChanged= true; }
    inline void setName(const std::string &name) { _name = name; }

    //!< Query functions return values based on the higher-order values present as indicated for setter functions.
    bool hasDirection(Direction direction) const;
    unsigned int numChannels() const;
    bool hasRange(unsigned int range) const;
    bool hasAref(Aref aref) const;

    //!< Get range information of any range assuming all other channel parameters as present in the object, e.g. for UI queries
    double rangeMin(unsigned int range) const;
    double rangeMax(unsigned int range) const;
    std::string rangeUnit(unsigned int range) const;

    //!< Getters
    inline Channel::Direction direction() const { return _type; }
    inline const inputSpec &waveform() const { return *_waveform; }
    double offset() const;
    double conversionFactor() const;
    int device() const;
    unsigned int channel() const;
    unsigned int range() const;
    Aref aref() const;
    inline const std::string &name() const { return _name; }
    inline int ID() const { return _ID; }
    int offsetSource() const;

    //!< Conversion between low-level and physical-unit values
    lsampl_t convert(double voltage) const;
    double convert(lsampl_t sample) const;

// ---------- Runtime interface --------------------------------
    void flush(); //!< Flush the read queue
    double nextSample(); //!< Get the next sample from the read queue
    void readOffset(); //!< Set the offset based on a reading from the channel indicated to Channel::setOffsetSource

// ----- Low-level interface -------------------------------------
    inline bool read(double &sample, bool hint = false) const {
        lsampl_t s; bool r = read(s, hint); sample = convert(s); return r;
    }
    inline bool write(double sample) const {
        return write(convert(sample));
    }
    bool read(lsampl_t &sample, bool hint = false) const; //!< Acquires a sample from this channel
    bool write(lsampl_t sample) const; //!< Writes the specified sample to analog out
    void put(lsampl_t &sample); //!< Adds @a sample to the input channel's read queue

    /** Query & reset - caution, resetting may cause output to go out of sync with the set waveform.
     * Calling this function from outside of waveform consumers is not recommended.
     **/
    inline bool waveformChanged() {
        bool c = *_waveformChanged; *_waveformChanged = false; return c;
    }

private:
    Channel::Direction _type;
    std::shared_ptr<inputSpec> _waveform;
    std::shared_ptr<bool> _waveformChanged;
    std::string _name;

    static int nextID;
    int _ID;

    class Impl;
    std::unique_ptr<Impl> pImpl;
};

#endif // CHANNEL_H
