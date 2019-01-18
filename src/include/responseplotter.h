#ifndef RESPONSEPLOTTER_H
#define RESPONSEPLOTTER_H

#include <QWidget>
#include <QTimer>
#include "queue.h"
#include "types.h"
#include "daq.h"

namespace Ui {
class ResponsePlotter;
}

class ResponsePlotter : public QWidget
{
    Q_OBJECT

public:
    explicit ResponsePlotter(QWidget *parent = 0);
    ~ResponsePlotter();

    RTMaybe::Queue<DataPoint> qI, qV, qV2, qO;
    const bool *VC = nullptr;

    //! Pass a DAQ to read I and V from directly, rather than using the DataPoint queues. The output trace is left blank.
    //! Time is deduced from DAQ::samplingDt(), with t=0 for the first sample after a call to clear().
    void setDAQ(DAQ *daq);

public slots:
    void start();
    void stop();
    void clear();

protected slots:
    void replot();

private:
    Ui::ResponsePlotter *ui;
    QTimer dataTimer;
    DAQ *daq = nullptr;
    size_t iT = 0;
};

#endif // RESPONSEPLOTTER_H
