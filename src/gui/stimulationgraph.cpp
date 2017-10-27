#include "stimulationgraph.h"

StimulationGraph::StimulationGraph(QCPAxis *keyAxis, QCPAxis *valueAxis, Stimulation stim, bool omitTail) :
    QCPGraph(keyAxis, valueAxis),
    m_stim(std::move(stim))
{
    scalar tLast;
    if ( omitTail )
        tLast = m_stim.tObsEnd;
    else
        tLast = m_stim.duration;

    bool complete = false;
    QVector<double> t, V;
    t.push_back(0);
    V.push_back(m_stim.baseV);
    scalar prevV = m_stim.baseV;
    scalar prevT = 0;
    for ( Stimulation::Step s : m_stim ) {
        if ( s.t > tLast ) {
            t.push_back(tLast);
            V.push_back(s.ramp ? prevV + (s.V-prevV)*(tLast-prevT)/(s.t-prevT) : prevV);
            complete = true;
            break;
        }

        if ( !s.ramp ) {
            t.push_back(s.t);
            V.push_back(prevV);
        }
        t.push_back(s.t);
        V.push_back(s.V);
        prevV = s.V;
        prevT = s.t;
    }
    if ( !complete ) {
        t.push_back(tLast);
        V.push_back(prevV);
    }
    setData(t, V, true);
}

StimulationGraph::~StimulationGraph() {}

void StimulationGraph::draw(QCPPainter *painter)
{
    QBrush brush = Qt::NoBrush;
    using std::swap;
    swap(brush, mBrush);
    QCPGraph::draw(painter);
    swap(brush, mBrush);
    painter->setBrush(mBrush);
    painter->setPen(Qt::NoPen);
    painter->drawRect(QRectF(QPointF( // top left
                                  keyAxis()->coordToPixel(m_stim.tObsBegin),
                                  valueAxis()->coordToPixel(valueAxis()->range().upper)),
                            QPointF( // bottom right
                                  keyAxis()->coordToPixel(m_stim.tObsEnd),
                                  valueAxis()->coordToPixel(valueAxis()->range().lower))));
}
