#ifndef STIMULATIONGRAPH_H
#define STIMULATIONGRAPH_H

#include "qcustomplot.h"
#include "types.h"

class StimulationGraph: public QCPGraph
{
    Q_OBJECT
public:
    StimulationGraph(QCPAxis *keyAxis, QCPAxis *valueAxis, Stimulation stim, bool omitTail = false);

    virtual ~StimulationGraph();

protected:
    Stimulation m_stim;

    virtual void draw(QCPPainter *painter);
};

#endif // STIMULATIONGRAPH_H
