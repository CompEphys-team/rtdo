#ifndef WAVEGENDIALOG_H
#define WAVEGENDIALOG_H

#include <QDialog>
#include <QThread>
#include <QButtonGroup>
#include "qcustomplot.h"
#include "wavegen.h"

namespace Ui {
class WavegenDialog;
}

class WavegenDialog : public QDialog
{
    Q_OBJECT

public:
    explicit WavegenDialog(MetaModel &model, QThread *thread, QWidget *parent = 0);
    ~WavegenDialog();

signals:
    void permute();
    void adjustSigmas();
    void search(int param);

private slots:
    void end(int);
    void startedSearch(int);
    void searchTick(int);

    void replot();
    void setPlotMinMaxSteps(int);

private:
    Ui::WavegenDialog *ui;
    QThread *thread;

    MetaModel &model;
    WavegenLibrary lib;
    Wavegen *wg;

    std::list<QString> actions;

    bool abort;

    QButtonGroup *groupx, *groupy;
    std::vector<QDoubleSpinBox*> mins, maxes;

    QCPColorMap *colorMap;

    void initWG();
    void initPlotControls();
    void refreshPlotControls();
};

#endif // WAVEGENDIALOG_H
