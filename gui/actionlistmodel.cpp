/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-04-13

--------------------------------------------------------------------------*/
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
    case ModelStimulate:
        as.handle = module->push(label(as), [=](int h) {
            module->vclamp->log()->score();
            auto sorted = module->vclamp->log()->sort(backlog::BacklogVirtual::RankScore, true);
            if ( sorted.empty() ) {
                cerr << "No models on record" << endl;
                return;
            }

            ofstream tf(module->outdir + "/" + to_string(h) + ".simtrace");
            const backlog::LogEntry *entry = sorted.front();

            tf << "# Traces from model:" << endl;
            vector<const backlog::LogEntry*> tmp(1, entry);
            write_backlog(tf, tmp, false);

            vector<vector<double>> traces = module->vclamp->stimulateModel(entry->idx);
            tf << endl << endl << "Time";
            for ( size_t i = 0; i < traces.size(); i++ ) {
                tf << '\t' << "Stimulation_" << i;
            }
            tf << endl;
            for ( size_t n = traces.size(), i = 0; n > 0; i++ ) { // Output data in columns, pad unequal lengths with '.' for gnuplotting
                n = traces.size();
                tf << (i * config->io.dt);
                for ( vector<double> &it : traces ) {
                    tf << '\t';
                    if ( it.size() <= i ) {
                        tf << '.';
                        --n;
                    } else {
                        tf << it.at(i);
                    }
                }
                tf << endl;
            }
        });
        break;
    case TracesDrop:
        as.handle = module->push(label(as), [=](int) {
            module->vclamp->data()->clear();
        });
        break;
    case TracesSave:
        as.handle = module->push(label(as), [=](int h) {
            ofstream tf(module->outdir + "/" + to_string(h) + ".traces");
            module->vclamp->data()->dump(tf);
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

bool ActionListModel::save(string filename)
{
    TiXmlDocument doc;
    doc.LinkEndChild(new TiXmlDeclaration("1.0", "", ""));
    TiXmlElement *root = new TiXmlElement("rtdoProtocol");
    doc.LinkEndChild(root);

    for ( auto it = actions.begin(); it != actions.end(); ++it ) {
        TiXmlElement *actionElement = new TiXmlElement("action");
        actionElement->SetAttribute("arg", it->arg);
        actionElement->LinkEndChild(new TiXmlText(std::to_string(static_cast<int>(it->action))));
        root->LinkEndChild(actionElement);
    }

    return doc.SaveFile(filename);
}

bool ActionListModel::load(string filename)
{
    TiXmlDocument doc;
    doc.LoadFile(filename);
    TiXmlHandle hDoc(&doc);
    TiXmlElement *root = hDoc.FirstChildElement("rtdoProtocol").Element();
    if ( !root )
        return false;

    for ( TiXmlElement *el = root->FirstChildElement("action"); el; el = el->NextSiblingElement("action") ) {
        Action action = static_cast<Action>(atoi(el->GetText()));
        int arg;
        el->QueryIntAttribute("arg", &arg);
        appendItem(action, arg);
    }
    return true;
}

QString ActionListModel::qlabel(ActionListModel::ActionStruct a)
{
    QString prefix("");
    if ( a.handle > 0 )
        prefix = QString("%1: ").arg(a.handle);
    switch ( a.action ) {
    case VCFrontload: return prefix + "Voltage clamp: Frontload stimulations, " + (a.arg ? QString("fitting") : QString("invariant"));
    case VCCycle: return prefix + "Voltage clamp: Cycle stimulations, " + (a.arg ? QString("fitting") : QString("invariant"));
    case VCRun: return prefix + "Voltage clamp: Fit models, " + (a.arg ? QString("%1 epochs").arg(a.arg) : "indefinitely");
    case ModelsSaveAll: return prefix + "Save all models";
    case ModelsSaveEval: return prefix + "Save evaluated models";
    case ModelStimulate: return prefix + "Stimulate best model";
    case TracesDrop: return prefix + "Drop traces";
    case TracesSave: return prefix + "Save traces";
    default: return prefix + "Unknown action";
    }
}

std::string ActionListModel::label(ActionStruct a)
{
    return qlabel(a).toStdString();
}

