#ifndef PARAMETERFITPLOTTER_H
#define PARAMETERFITPLOTTER_H

#include <QWidget>
#include "session.h"
#include "qcustomplot.h"
#include "colorbutton.h"

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
    std::vector<int> getSelectedRows();
    ColorButton *getGraphColorBtn(int row);
    ColorButton *getErrorColorBtn(int row);

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

    bool resizing, enslaved;

    std::vector<QColor> clipboard;
};

#endif // PARAMETERFITPLOTTER_H
