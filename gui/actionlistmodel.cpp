#include "actionlistmodel.h"
#include <fstream>
#include "run.h"

ActionListModel::ActionListModel(Module *module, QObject *parent) :
    QAbstractListModel(parent),
    module(module)
{
    connect(module, SIGNAL(complete(int)), this, SLOT(actionComplete(int)));
}

QVariant ActionListModel::data(const QModelIndex &index, int role) const
{
    if ( role == Qt::DisplayRole && index.row() >= 0 ) {
        return qlabel(actions.at(index.row()));
    }
    return QVariant();
}

int ActionListModel::rowCount(const QModelIndex &parent) const
{
    return actions.size();
}

void ActionListModel::appendItem(ActionListModel::Action a, int arg)
{
    ActionStruct as {a, arg, 0};
    switch ( a ) {
    case VCFrontload:
        as.handle = module->push(label(as), [=](int) {
            for ( unsigned int i = 0; i < config->vc.cacheSize && !module->vclamp->stopFlag; i++ )
                module->vclamp->cycle(arg);
        });
        break;
    case VCCycle:
        as.handle = module->push(label(as), [=](int) {
            module->vclamp->cycle(arg);
        });
        break;
    case VCRun:
        as.handle = module->push(label(as), [=](int) {
            module->vclamp->run(arg);
        });
        break;
    case ModelsSaveAll:
        as.handle = module->push(label(as), [=](int h) {
            ofstream logf(module->outdir + "/" + to_string(h) + "_modelsAll.log");
            module->vclamp->log()->score();
            write_backlog(logf, module->vclamp->log()->sort(backlog::BacklogVirtual::RankScore, true), false);
        });
        break;
    case ModelsSaveEval:
        as.handle = module->push(label(as), [=](int h) {
            ofstream logf(module->outdir + "/" + to_string(h) + "_modelsEval.log");
            module->vclamp->log()->score();
            write_backlog(logf, module->vclamp->log()->sort(backlog::BacklogVirtual::RankScore, true), true);
        });
        break;
    case TracesDrop:
        as.handle = module->push(label(as), [=](int h) {
            ofstream tf(module->outdir + "/" + to_string(h) + ".traces");
            module->vclamp->data()->dump(tf);
        });
        break;
    case TracesSave:
        as.handle = module->push(label(as), [=](int) {
            module->vclamp->data()->clear();
        });
        break;
    default:
        return;
    }

    int r = rowCount();
    beginInsertRows(QModelIndex(), r, r+1);
    actions.push_back(as);
    endInsertRows();
}

void ActionListModel::actionComplete(int handle)
{
    int i = 0;
    for ( auto it = actions.begin(); it != actions.end(); ++it, ++i ) {
        if ( it->handle == handle ) {
            beginRemoveRows(QModelIndex(), i, i);
            actions.erase(it);
            endRemoveRows();
            break;
        }
    }
}

void ActionListModel::removeItem(const QModelIndex &index)
{
    if ( !index.isValid() )
        return;

    beginRemoveRows(QModelIndex(), index.row(), index.row());
    auto it = actions.begin() + index.row();
    if ( module->erase(it->handle) ) {
        actions.erase(it);
    } else if ( index.row() == 0 ) {
        module->skip();
    }
    endRemoveRows();
}

void ActionListModel::clear()
{
    beginRemoveRows(QModelIndex(), 0, rowCount());
    actions.clear();
    endRemoveRows();
}

QString ActionListModel::qlabel(ActionListModel::ActionStruct a)
{
    QString prefix = QString("%1: ").arg(a.handle);
    switch ( a.action ) {
    case VCFrontload: return prefix + "Voltage clamp: Frontload stimulations, " + (a.arg ? QString("fitting") : QString("invariant"));
    case VCCycle: return prefix + "Voltage clamp: Cycle stimulations, " + (a.arg ? QString("fitting") : QString("invariant"));
    case VCRun: return prefix + "Voltage clamp: Fit models, " + (a.arg ? QString("%1 epochs").arg(a.arg) : "indefinitely");
    case ModelsSaveAll: return prefix + "Save all models";
    case ModelsSaveEval: return prefix + "Save evaluated models";
    case TracesDrop: return prefix + "Drop traces";
    case TracesSave: return prefix + "Save traces";
    default: return prefix + "Unknown action";
    }
}

std::string ActionListModel::label(ActionStruct a)
{
    return qlabel(a).toStdString();
}

