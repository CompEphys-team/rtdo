#include "stimulationgraph.h"

StimulationGraph::StimulationGraph(QCPAxis *keyAxis, QCPAxis *valueAxis, Stimulation stim, bool omitTail) :
    QCPGraph(keyAxis, valueAxis),
    m_stim(std::move(stim)),
    m_tail(!omitTail)
{
    recalculate();
}

void StimulationGraph::recalculate()
{
    scalar tLast;
    if ( m_tail )
        tLast = m_stim.duration;
    else if ( m_dt == 0 )
        tLast = m_stim.tObsEnd;
    else {
        size_t i;
        for ( i = 0; i < iObservations::maxObs && m_obs.stop[i] > 0; i++ )
            ;
        tLast = i ? m_obs.stop[i-1] * m_dt : 0;
    }

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

void StimulationGraph::setObservations(const iObservations &obs, double dt)
{
     m_obs = obs;
     m_dt = dt;
     recalculate();
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
    if ( m_dt == 0 ) {
        painter->drawRect(QRectF(QPointF( // top left
                                      keyAxis()->coordToPixel(m_stim.tObsBegin),
                                      valueAxis()->coordToPixel(valueAxis()->range().upper)),
                                QPointF( // bottom right
                                      keyAxis()->coordToPixel(m_stim.tObsEnd),
                                      valueAxis()->coordToPixel(valueAxis()->range().lower))));
    } else {
        for ( size_t i = 0; i < iObservations::maxObs && m_obs.stop[i] > 0; i++ ) {
            painter->drawRect(QRectF(QPointF( // top left
                                          keyAxis()->coordToPixel(m_obs.start[i] * m_dt),
                                          valueAxis()->coordToPixel(valueAxis()->range().upper)),
                                    QPointF( // bottom right
                                          keyAxis()->coordToPixel(m_obs.stop[i] * m_dt),
                                          valueAxis()->coordToPixel(valueAxis()->range().lower))));
        }
    }
}
