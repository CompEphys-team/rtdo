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


#ifndef STIMULATIONGRAPH_H
#define STIMULATIONGRAPH_H

#include "qcustomplot.h"
#include "types.h"

class StimulationGraph: public QCPGraph
{
    Q_OBJECT
public:
    StimulationGraph(QCPAxis *keyAxis, QCPAxis *valueAxis, Stimulation stim, bool omitTail = false);
    void setObservations(const iObservations &obs, double dt);

    virtual ~StimulationGraph();

protected:
    Stimulation m_stim;
    bool m_tail;

    iObservations m_obs = {{}, {}};
    double m_dt = 0;

    virtual void draw(QCPPainter *painter);
    void recalculate();
};

#endif // STIMULATIONGRAPH_H
