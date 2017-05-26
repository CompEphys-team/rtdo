#include "stimulationgraph.h"

StimulationGraph::StimulationGraph(QCPAxis *keyAxis, QCPAxis *valueAxis, Stimulation stim) :
    QCPGraph(keyAxis, valueAxis),
    m_stim(std::move(stim))
{
    QVector<double> t, V;
    t.push_back(0);
    V.push_back(m_stim.baseV);
    scalar prevV = m_stim.baseV;
    for ( Stimulation::Step s : m_stim ) {
        if ( !s.ramp ) {
            t.push_back(s.t);
            V.push_back(prevV);
        }
        t.push_back(s.t);
        V.push_back(s.V);
        prevV = s.V;
    }
    t.push_back(m_stim.duration);
    V.push_back(prevV);
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
