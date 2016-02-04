#ifndef RUNNER_H
#define RUNNER_H

#include <QObject>
#include "realtimethread.h"
#include "xmlmodel.h"

class Runner : public QObject
{
    Q_OBJECT
public:
    explicit inline Runner(QObject *parent = 0) : QObject(parent), _running(false) {}
    inline bool running() const { return _running; }

signals:
    void processCompleted(bool successfully);

public slots:
    virtual bool start() = 0;
    inline virtual bool stop() { return false; }

protected:
    virtual void *launch() = 0;
    static void *launchStatic(void *_this);
    friend class RealtimeThread::Impl;

    std::unique_ptr<RealtimeThread> t;

    bool _running;
};


class CompileRunner : public Runner
{
    Q_OBJECT
public:
    explicit CompileRunner(XMLModel::outputType type = XMLModel::WaveGen, QObject *parent = 0);

    inline void setType(XMLModel::outputType type) { _type = type; }
    inline XMLModel::outputType type() const { return _type; }

public slots:
    bool start();

protected:
    inline void *launch() { return 0; }

private:
    XMLModel::outputType _type;
};


class VClampRunner : public Runner
{
    Q_OBJECT
public:
    explicit inline VClampRunner(QObject *parent = 0) : Runner(parent) {}

public slots:
    bool start();
    bool stop();

protected:
    void *launch();

private:
    bool _stop;
};


class WaveGenRunner : public Runner
{
    Q_OBJECT
public:
    explicit inline WaveGenRunner(QObject *parent = 0) : Runner(parent) {}

public slots:
    bool start();
    bool stop();

protected:
    void *launch();

private:
    bool _stop;
};

#endif // RUNNER_H