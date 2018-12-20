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
