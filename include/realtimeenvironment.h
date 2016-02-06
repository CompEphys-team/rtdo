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

    //!< Reset and start the analog read/write process in RT mode; reset the simulation in simulation mode
    void sync();

    //!< Stop the analog read/write process in RT mode; has no function in simulation mode
    void pause();

    //!< Set the supersampling rate for RT mode (i.e. the number of reads from analog in averaged to produce one sample)
    void setSupersamplingRate(int acquisitionsPerSample);

    //!< Set the timestep of the simulation. Keep this strictly equal to DT in the GeNN code. Has no function in RT mode
    void setDT(double dt);

    //!< Set waveform to be used by both the designated analog out channel and the simulator
    void setWaveform(const inputSpec &i);

    //!< Get the next sample in the read queue of the specified channel in RT mode, or from the simulator in simulation mode
    double nextSample(int channelIndex = 0);

    /** Set variable and parameter values for use in the simulator. Note that variable values are altered during simulation.
     * Since the simulator precision used is not affected by these functions, it is highly recommended to pass "scalar" type.
     * Failure to do so may result in undefined behaviour (i.e., previous or unset values are passed to the simulator).
     **/
    void setSimulatorVariables(double *vars);
    void setSimulatorVariables(float *vars);
    void setSimulatorParameters(double *params);
    void setSimulatorParameters(float *params);

    /** Use @a c starting from next call to @fn RealtimeEnvironment::sync.
     * Note: At any given time, only one output channel can be active. Previously set output channels are therefore discarded.
     **/
    void addChannel(Channel &c);

    //!< Clear any channels previously set by addChannel
    void clearChannels();

    /** Set the function to use as a simulator. Float and double versions are mutually exclusive, the latest set will be used.
     * If set from within GeNN code, the scalar version is acceptable and indeed preferable, see comment to @fn setSimulatorVariables.
     **/
    void setSimulator(double (*fn)(double*, double*, double));
    void setSimulator(float (*fn)(float *, float *, float));

    /** Switch between simulation and RT mode. Returns true if successful.
     * In non-RT builds, this function has no effect and returns false, as the environment is always in simulation mode.
     **/
    bool useSimulator(bool);

    /** Couple a variable (usually @var clampGainHH) to the configured clamp gain. The indicated value is updated at each @fn sync().
     * This is called from within @fn rtdo_setup_bridge and need not be repeated elsewhere.
     * The native type of @a param should be scalar*.
     **/
    void setClampGainParameter(void *param);

    //!< Get the configured clamp gain value
    double getClampGain();

    //!< Opens and returns a comedi device in hard/soft realtime [RT build only]
    struct comedi_t_struct *getDevice(int deviceno, bool RT);

    //!< Returns the advertised name of the given comedi device [RT build only]
    std::string getDeviceName(int deviceno);

    //!< Finds the first subdevice of the given type [RT build only]
    unsigned int getSubdevice(int deviceno, Channel::Direction type);

    /** Turns reporting about free time or overruns in the analog input thread on or off. By default, reporting is off.
     * Enable to diagnose supersampling ability, but preferably disable for production as it takes a considerable amount
     * of time to process during sync.
     **/
    void setIdleTimeReporting(bool);
    bool idleTimeReporting() const;

private:
    RealtimeEnvironment();

    class Simulator;
    std::unique_ptr<Simulator> sImpl;

    class Impl;
    std::unique_ptr<Impl> pImpl;

    bool _useSim;

    void *clampGainParam;
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
