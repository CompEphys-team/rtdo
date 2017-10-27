#ifndef STIMULATIONPLOTTER_H
#define STIMULATIONPLOTTER_H

#include <QWidget>
#include "session.h"
#include "qcustomplot.h"
#include "stimulationgraph.h"

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
    void resizePanel();
    void updateSources();
    void replot();
    void resizeEvent(QResizeEvent *event);

private slots:
    void on_pdf_clicked();

private:
    Ui::StimulationPlotter *ui;
    Session *session;

    std::vector<StimulationGraph*> graphs;
    std::vector<QColor> colors;

    bool rebuilding;

    bool enslaved;
    bool single;
    WaveSource source;
    Stimulation stim;
};

#endif // STIMULATIONPLOTTER_H
