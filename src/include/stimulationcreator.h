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
        iStimulation stim;
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

    void setupPlot();
    void setupTraces();
    void addTrace(Trace &trace, int idx);

protected slots:
    void updateSources();
    void copySource();
    void setNStims(int n);
    void setLimits();
    void setStimulation();
    void setNSteps(int n);
    void setObservations();
    void updateStimulation();
    void redraw();
    void diagnose();
    void clustering();
    void traceEdited(QTableWidgetItem *item);

private slots:
    void on_paramTrace_clicked();

    void on_pdf_clicked();

    void on_cl_magic_clicked();

private:
    Ui::StimulationCreator *ui;
    Session &session;
    std::vector<iStimulation> stims;
    std::vector<iStimulation>::iterator stim;
    std::vector<iObservations> observations;
    std::vector<iObservations>::iterator obsIt;
    iStimulation stimCopy;
    iObservations obsCopy;
    bool loadingStims, updatingStim;

    std::vector<Trace> traces;
    DAQ *simulator;
    enum {SrcBase, SrcFit, SrcManual, SrcRec} paramsSrc;

    UniversalLibrary *lib = nullptr;
    std::vector<std::vector<scalar*>> pDelta;
};

#endif // STIMULATIONCREATOR_H
