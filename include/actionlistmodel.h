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
        VCFrontload,
        VCCycle,
        VCRun,
        ModelsSaveAll,
        ModelsSaveEval,
        TracesSave,
        TracesDrop
    };

public slots:
    void appendItem(Action a, int arg = 0);
    void actionComplete(int handle);
    void removeItem(const QModelIndex &index);
    void clear();

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
