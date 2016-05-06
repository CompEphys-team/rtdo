/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-04-13

--------------------------------------------------------------------------*/
#ifndef ACTIONLISTMODEL_H
#define ACTIONLISTMODEL_H

#include <QAbstractListModel>
#include "module.h"
#include <deque>

class ActionListModel : public QAbstractListModel
{
    Q_OBJECT
public:
    ActionListModel(QObject *parent = 0);

    QVariant data(const QModelIndex & index, int role = Qt::DisplayRole) const;
    int rowCount(const QModelIndex & parent = QModelIndex()) const;

    enum Action {
        Invalid = 0,

        VCFrontload = 1,
        VCCycle = 2,
        VCRun = 3,

        ModelsSaveAll = 11,
        ModelsSaveEval = 12,
        ModelStimulate = 13,

        TracesSave = 21,
        TracesDrop = 22,

        ParamFix = 31,

        SigmaAdjust = 41,
        NoveltySearch = 42,
        OptimiseWave = 43,
        OptimiseAllWaves = 44,

        VCWaveCurrents = 51
    };

public slots:
    virtual void appendItem(Action a, int arg = 0, double darg = 0.0) = 0;
    void actionComplete(int handle);
    virtual void removeItem(const QModelIndex &index) = 0;
    void clear();

    bool save(std::string filename);
    bool load(std::string filename);

protected:
    struct ActionStruct {
        Action action;
        int arg;
        int handle;
        double darg;
    };

    std::deque<ActionStruct> actions;

    static QString qlabel(ActionStruct a);
    static std::string label(ActionStruct a);
};

template <class T>
class Protocol : public ActionListModel
{
public:
    Protocol(Module<T> *module, QObject *parent = 0);
    void appendItem(Action a, int arg = 0, double darg = 0.0);
    void removeItem(const QModelIndex &index);

private:
    Module<T> *module;
};

#endif // ACTIONLISTMODEL_H
