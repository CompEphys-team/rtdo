#ifndef STIMULATIONPLOTTER_H
#define STIMULATIONPLOTTER_H

#include <QWidget>
#include "session.h"
#include "qcustomplot.h"

namespace Ui {
class StimulationPlotter;
}

class StimulationPlotter : public QWidget
{
    Q_OBJECT

public:
    explicit StimulationPlotter(QWidget *parent = 0);
    StimulationPlotter(Session &session, QWidget *parent = 0);
    ~StimulationPlotter();

    void init(Session *session);
    void setSource(WaveSource src);
    void setStimulation(Stimulation src);
    void clear();

protected:
    void updateColor(size_t idx, bool replot);

protected slots:
    void setColumnCount(int n);
    void resizeTableRows(int, int, int size);
    void updateSources();
    void replot();

private:
    Ui::StimulationPlotter *ui;
    Session *session;

    std::vector<QCustomPlot*> plots;
    std::vector<QColor> colors;

    bool resizing;
    bool rebuilding;

    bool enslaved;
    bool single;
    WaveSource source;
    Stimulation stim;
};

#endif // STIMULATIONPLOTTER_H
