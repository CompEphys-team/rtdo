#ifndef WAVEGENPROGRESSPLOTTER_H
#define WAVEGENPROGRESSPLOTTER_H

#include <QWidget>
#include "session.h"
#include "qcustomplot.h"

namespace Ui {
class WavegenProgressPlotter;
}

class QCheckBox;
struct AbstractGraphProxy
{
    AbstractGraphProxy(QColor color, QCheckBox *cb);
    virtual ~AbstractGraphProxy() {}

    virtual void populate(double keyFactor) = 0;
    virtual void extend() = 0;

    QColor color;
    QCheckBox *cb;
    QCPAxis *xAxis = 0, *yAxis = 0;
    const Wavegen::Archive *archive;
    QSharedPointer<QCPGraphDataContainer> dataPtr;
    double factor;
};

class WavegenProgressPlotter : public QWidget
{
    Q_OBJECT

public:
    explicit WavegenProgressPlotter(QWidget *parent = 0);
    void init(Session &session);
    ~WavegenProgressPlotter();

protected slots:
    void updateArchives();
    void searchTick(int);
    void replot();

private:
    Ui::WavegenProgressPlotter *ui;
    Session *session;
    bool inProgress;
    std::vector<AbstractGraphProxy*> proxies;
};

#endif // WAVEGENPROGRESSPLOTTER_H
