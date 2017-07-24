#ifndef RESPONSEPLOTTER_H
#define RESPONSEPLOTTER_H

#include <QWidget>
#include <QTimer>
#include "queue.h"
#include "types.h"

namespace Ui {
class ResponsePlotter;
}

class ResponsePlotter : public QWidget
{
    Q_OBJECT

public:
    explicit ResponsePlotter(QWidget *parent = 0);
    ~ResponsePlotter();

    RTMaybe::Queue<DataPoint> qI, qV, qO;

public slots:
    void clear();

protected slots:
    void replot();

private:
    Ui::ResponsePlotter *ui;
    QTimer dataTimer;
};

#endif // RESPONSEPLOTTER_H
