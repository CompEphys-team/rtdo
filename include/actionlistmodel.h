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
    ActionListModel(Module *module, QObject *parent = 0);

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
        TracesDrop = 22
    };

public slots:
    void appendItem(Action a, int arg = 0);
    void actionComplete(int handle);
    void removeItem(const QModelIndex &index);
    void clear();

    bool save(std::string filename);
    bool load(std::string filename);

private:
    Module *module;

    struct ActionStruct {
        Action action;
        int arg;
        int handle;
    };

    std::deque<ActionStruct> actions;

    static QString qlabel(ActionStruct a);
    static std::string label(ActionStruct a);
};

#endif // ACTIONLISTMODEL_H
