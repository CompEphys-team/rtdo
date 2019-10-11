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


#ifndef CANNEDDAQ_H
#define CANNEDDAQ_H

#include "daq.h"
#include <vector>
#include "wavesubsets.h"

class CannedDAQ : public DAQ
{
public:
    CannedDAQ(Session &s, const Settings &settings);
    ~CannedDAQ() = default;

    double getAdjustableParam(size_t);
    void setAdjustableParam(size_t, double) {}
    int throttledFor(const Stimulation &) { return 0; }
    void run(Stimulation s, double settleDuration = 0);
    void next();
    void reset();

    bool setRecord(std::vector<Stimulation> stims, QString record, bool readData = true);
    std::vector<QuotedString> channelNames;

    struct ChannelAssociation {
        int Iidx = 0, Vidx = -1, V2idx = -1;
        double Iscale = 1, Vscale = 1, V2scale = 1;
    };
    ChannelAssociation assoc;

    void getSampleNumbers(const std::vector<Stimulation> &stims, double dt,
                          int *nTotal, int *nBuffer = nullptr, int *nSamples = nullptr);

    double variance = 0;

protected:
    int prepareRecords(std::vector<Stimulation> stims);

    struct Record
    {
        Stimulation stim;
        int nBuffer, nTotal;
        std::vector<double> I, V, V2;
    };
    std::vector<Record> records;
    size_t currentRecord;
    int recordIndex;
    double settleDuration;

    void getReferenceParams(QString record);
    std::vector<double> ref_params;
};

#endif // CANNEDDAQ_H
