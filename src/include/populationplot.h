#ifndef POPULATIONPLOT_H
#define POPULATIONPLOT_H

#include <QWidget>
#include "session.h"
#include "qcustomplot.h"

namespace Ui {
class PopulationPlot;
}

class PopulationPlot : public QWidget
{
    Q_OBJECT

public:
    explicit PopulationPlot(QWidget *parent = 0);
    PopulationPlot(Session &session, QWidget *parent = 0);
    ~PopulationPlot();

    void init(Session *session, bool enslave);
    void clear();

public slots:
    void replot();
    void updateCombos();

protected slots:
    void resizeEvent(QResizeEvent *event);
    void resizePanel();
    void clearPlotLayout();
    void buildPlotLayout();
    void xRangeChanged(QCPRange);

private:
    Ui::PopulationPlot *ui;
    Session *session;
    UniversalLibrary *lib = nullptr;

    std::vector<QCPAxisRect*> axRects;

    bool enslaved = false;
};

#endif // POPULATIONPLOT_H
