#ifndef CALIBRATOR_H
#define CALIBRATOR_H

#include <QObject>
#include "session.h"

class Calibrator : public QObject
{
    Q_OBJECT
public:
    Calibrator(Session &s, QObject *parent = nullptr);

public slots:
    void zeroV1(DAQData p);
    void zeroV2(DAQData p);
    void zeroIin(DAQData p);
    void zeroVout(DAQData p);

signals:
    void zeroingV1(bool done);
    void zeroingV2(bool done);
    void zeroingIin(bool done);
    void zeroingVout(bool done);

protected:
    Session &session;
};

#endif // CALIBRATOR_H
