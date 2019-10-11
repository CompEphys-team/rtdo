/** RTDO: Closed loop neuron model fitting
 *  Copyright (C) 2019 Felix Kern <kernfel+github@gmail.com>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
**/


#ifndef COMEDIDAQ_H
#define COMEDIDAQ_H

#include "daq.h"
#include "thread.h"
#include "conditionvariable.h"
#include "queue.h"
#include "comediconverter.h"
#include <QTime>

namespace RTMaybe {
    class ComediDAQ;
}

class RTMaybe::ComediDAQ : public DAQ
{
public:
    ComediDAQ(Session &session, const Settings &settings);
    ~ComediDAQ();

    inline double getAdjustableParam(size_t) { return 0; }
    inline void setAdjustableParam(size_t, double) {}
    int throttledFor(const Stimulation &s);
    void run(Stimulation s, double settleDuration = 0);
    void next();
    void reset();

    void scope(int qSize); //!< Acquire in endless mode, running until reset() is called. AO is set to 0.
                           //!  Set @a qSize to a sufficient number of samples for your next() polling frequency.
                           //! DAQ::samplesRemaining is populated with the number of samples that remain in the queue upon calling next().

    inline int nSamples() const { return currentStim.duration / samplingDt() + (p.filter.active ? p.filter.width : 0); }

protected:
    bool live;
    bool endless = false;
    RTMaybe::ConditionVariable ready, set, go, finish;
    RTMaybe::Queue<lsampl_t> qI, qV, qV2;
    RTMaybe::Thread t;

    ComediConverter conI, conV, conV2, conVC, conCC;

    QTime wallclock;

    void dispatch(int qSize);

    static void *launchStatic(void *);
    void *launch();

    template <typename aisampl_t, typename aosampl_t>
    void acquisitionLoop(void *dev, int aidev, int aodev);
};

#endif // COMEDIDAQ_H
