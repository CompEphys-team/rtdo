#ifndef PARAMETERFITPLOTTER_H
#define PARAMETERFITPLOTTER_H

#include <QWidget>
#include "session.h"
#include "qcustomplot.h"

namespace Ui {
class ParameterFitPlotter;
}

class ParameterFitPlotter : public QWidget
{
    Q_OBJECT

public:
    explicit ParameterFitPlotter(QWidget *parent = 0);
    ParameterFitPlotter(Session &session, QWidget *parent = 0);
    ~ParameterFitPlotter();

    void init(Session *session, bool enslave);
    void clear();

protected:
    void styleErrorGraph(QCPGraph*);

private slots:
    void setColumnCount(int n);
    void resizeTableRows(int, int, int size);
    void updateFits();

    void replot();
    void progress(quint32);
    void rangeChanged(QCPRange range);
    void errorRangeChanged(QCPRange range);

private:
    Ui::ParameterFitPlotter *ui;
    Session *session;

    std::vector<QCustomPlot*> plots;

    bool resizing;
};

#endif // PARAMETERFITPLOTTER_H
