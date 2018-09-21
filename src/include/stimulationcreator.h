#ifndef STIMULATIONCREATOR_H
#define STIMULATIONCREATOR_H

#include <QWidget>
#include "session.h"
#include "qcustomplot.h"

namespace Ui {
class StimulationCreator;
}

class StimulationCreator : public QWidget
{
    Q_OBJECT

    struct Trace
    {
        Stimulation stim;
        QString label;
        std::vector<double> params;
        QVector<double> current, voltage;
        double dt;
        QCPGraph *gI, *gV;
    };

public:
    explicit StimulationCreator(Session &session, QWidget *parent = 0);
    ~StimulationCreator();

protected:
    void makeHidable(QCPGraph *g);

    void setupTraces();
    void addTrace(Trace &trace, int idx);

protected slots:
    void updateSources();
    void copySource();
    void setNStims(int n);
    void setLimits();
    void setStimulation();
    void setNSteps(int n);
    void updateStimulation();
    void redraw();
    void diagnose();
    void clustering();
    void traceEdited(QTableWidgetItem *item);

private slots:
    void on_paramTrace_clicked();

private:
    Ui::StimulationCreator *ui;
    Session &session;
    std::vector<Stimulation> stims;
    std::vector<Stimulation>::iterator stim;
    Stimulation stimCopy;
    bool loadingStims, updatingStim;

    std::vector<Trace> traces;
    DAQ *simulator;
    enum {SrcBase, SrcFit, SrcManual, SrcRec} paramsSrc;
    bool addingTrace = false;
};

#endif // STIMULATIONCREATOR_H
