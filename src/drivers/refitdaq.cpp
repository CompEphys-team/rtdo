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

#include "refitdaq.h"
#include "session.h"
#include "gafitter.h"
#include <exception>

RefitDAQ::RefitDAQ(Session &s, const Settings &settings, const GAFitter::Output &fit) :
    DAQ(s, settings),
    session(&s),
    fit(fit),
    trace_dt(s.runData(fit.resultIndex).dt)
{
    int n = s.gaFitterSettings(fit.resultIndex).cl_nStims;
    stims.reserve(fit.epochs * n);
    traces.reserve(fit.epochs * n);
    QFile file(QString("%1.traces.bin").arg(s.resultFilePath(fit.resultIndex)));
    QDataStream is;
    s.gaFitter().openLoadStream(file, is);

    int i = 0;
    while ( !is.atEnd() ) {
        iStimulation I;
        qint32 nSettleSamples;
        is >> I >> nSettleSamples;
        traces.push_back(std::vector<double>(nSettleSamples + I.duration));
        for ( int t = 0; t < nSettleSamples + I.duration; t++ )
            is >> traces[i][t];
        stims[i] = I;
        settleDur[i] = nSettleSamples;
    }
}

double RefitDAQ::getAdjustableParam(size_t i)
{
    return fit.targets[i];
}

void RefitDAQ::run(Stimulation s, double settleDuration)
{
    iStimulation I(s, trace_dt);
    trace_idx = std::min(trace_idx+1, stims.size()-1);
    if ( !(I == stims[trace_idx]) ) {
        for ( trace_idx = 0; trace_idx < stims.size(); trace_idx++ )
            if ( I == stims[trace_idx] )
                break;
        if ( trace_idx == stims.size() ) {
            trace_idx = 0;
            std::cerr << "Error: Stim not found in record: " << s << std::endl;
        }
    }

    sampleIdx = 0;
    int nReqSettleSamples = int(settleDuration/trace_dt);
    if ( nReqSettleSamples > settleDur[trace_idx] )
        nSkipSamples = nReqSettleSamples - settleDur[trace_idx];
    else
        sampleIdx = settleDur[trace_idx] - nReqSettleSamples;
}

void RefitDAQ::next()
{
    voltage = traces[trace_idx][sampleIdx];
    if ( nSkipSamples-- <= 0 )
        ++sampleIdx;
}
