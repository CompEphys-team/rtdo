#ifndef STIMULATIONCREATOR_H
#define STIMULATIONCREATOR_H

#include <QWidget>
#include "session.h"

namespace Ui {
class StimulationCreator;
}

class StimulationCreator : public QWidget
{
    Q_OBJECT

public:
    explicit StimulationCreator(Session &session, QWidget *parent = 0);
    ~StimulationCreator();

protected slots:
    void updateSources();
    void copySource();
    void setNStims(int n);
    void setLimits();
    void setStimulation();
    void setNSteps(int n);
    void updateStimulation();
    void redraw();

private:
    Ui::StimulationCreator *ui;
    Session &session;
    std::vector<Stimulation> stims;
    std::vector<Stimulation>::iterator stim;
    Stimulation stimCopy;
    bool loadingStims, updatingStim;
};

#endif // STIMULATIONCREATOR_H
