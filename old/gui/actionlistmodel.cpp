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

ActionListModel::ActionListModel(QObject *parent) :
    QAbstractListModel(parent)
{}

template <class T>
Protocol<T>::Protocol(Module<T> *module, QObject *parent) :
    ActionListModel(parent),
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

template <>
void Protocol<Experiment>::appendItem(Action a, int arg, double darg)
{
    ActionStruct as {a, arg, 0, darg};
    switch ( a ) {
    case VCFrontload:
        as.handle = module->push(label(as), [=](int) {
            for ( unsigned int i = 0; i < config->vc.cacheSize && !module->obj->stopFlag; i++ )
                module->obj->cycle(arg);
        });
        break;
    case VCCycle:
        as.handle = module->push(label(as), [=](int) {
            module->obj->cycle(arg);
        });
        break;
    case VCRun:
        as.handle = module->push(label(as), [=](int) {
            module->obj->run(arg);
        });
        break;
    case ModelsSaveAll:
        as.handle = module->push(label(as), [=](int h) {
            ofstream logf(module->outdir + "/" + to_string(h) + "_modelsAll.log");
            module->obj->log()->score();
            write_backlog(logf, module->obj->log()->sort(backlog::BacklogVirtual::RankScore, true), false);
        });
        break;
    case ModelsSaveEval:
        as.handle = module->push(label(as), [=](int h) {
            ofstream logf(module->outdir + "/" + to_string(h) + "_modelsEval.log");
            module->obj->log()->score();
            write_backlog(logf, module->obj->log()->sort(backlog::BacklogVirtual::RankScore, true), true);
        });
        break;
    case ModelStimulate:
        as.handle = module->push(label(as), [=](int h) {
            module->obj->log()->score();
            auto sorted = module->obj->log()->sort(backlog::BacklogVirtual::RankScore, true);
            if ( sorted.empty() ) {
                cerr << "No models on record" << endl;
                return;
            }

            ofstream tf(module->outdir + "/" + to_string(h) + ".simtrace");
            const backlog::LogEntry *entry = sorted.front();

            vector<const backlog::LogEntry*> tmp(1, entry);
            write_backlog(tf, tmp, false);

            vector<vector<double>> traces = module->obj->stimulateModel(entry->idx);
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
            module->obj->data()->clear();
        });
        break;
    case TracesSave:
        as.handle = module->push(label(as), [=](int h) {
            ofstream tf(module->outdir + "/" + to_string(h) + ".traces");
            module->obj->data()->dump(tf);
        });
        break;
    case ParamFix:
        as.handle = module->push(label(as), [=](int) {
            module->obj->fixParameter(as.arg, as.darg);
        });
        break;
    default:
        cerr << "Invalid action for this protocol type: " << a << endl;
        return;
    }

    int r = rowCount();
    beginInsertRows(QModelIndex(), r, r+1);
    actions.push_back(as);
    endInsertRows();
}

template<>
void Protocol<WavegenNSVirtual>::appendItem(Action a, int arg, double darg)
{
    ActionStruct as {a, arg, 0, darg};
    switch ( a ) {
    case SigmaAdjust:
        as.handle = module->push(label(as), [=](int) {
            module->obj->adjustSigmas();
        });
        break;
    case NoveltySearch:
        as.handle = module->push(label(as), [=](int) {
            module->obj->noveltySearch();
        });
        break;
    case OptimiseWave:
        as.handle = module->push(label(as), [=](int h) {
            ofstream wf(module->outdir + "/" + to_string(h) + "_" + config->model.obj->adjustableParams().at(as.arg).name + ".stim");
            ofstream cf(module->outdir + "/" + to_string(h) + "_" + config->model.obj->adjustableParams().at(as.arg).name + "currents.log");
            module->obj->optimise(as.arg, wf, cf);
        });
        break;
    case OptimiseAllWaves:
        as.handle = module->push(label(as), [=](int h) {
            ofstream wf(module->outdir + "/" + to_string(h) + ".stim");
            ofstream cf(module->outdir + "/" + to_string(h) + "_currents.log");
            module->obj->optimiseAll(wf, cf);
        });
        break;
    case VCWaveCurrents:
        cerr << "VCWaveCurrents NYI" << endl;
    default:
        cerr << "Invalid action for this protocol type: " << a << endl;
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

template <class T>
void Protocol<T>::removeItem(const QModelIndex &index)
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
        actionElement->SetDoubleAttribute("darg", it->darg);
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
        int arg = 0;
        double darg = 0.0;
        el->QueryIntAttribute("arg", &arg);
        el->QueryDoubleAttribute("darg", &darg);
        appendItem(action, arg, darg);
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
    case ParamFix: return prefix + QString("Fix parameter %1 to %2")\
                .arg(QString::fromStdString(config->model.obj->adjustableParams().at(a.arg).name))\
                .arg(a.darg);

    case SigmaAdjust: return prefix + "Adjust parameter sigmas";
    case NoveltySearch: return prefix + "Novelty search for waveforms";
    case OptimiseWave: return prefix + QString("Evolve waveforms for parameter %1")\
                .arg(QString::fromStdString(config->model.obj->adjustableParams().at(a.arg).name));
    case OptimiseAllWaves: return prefix + "Evolve waveforms for all parameters";
    case VCWaveCurrents: return "Calculate currents under voltage clamp waveform";
    default: return prefix + "Unknown action";
    }
}

std::string ActionListModel::label(ActionStruct a)
{
    return qlabel(a).toStdString();
}

template class Protocol<Experiment>;
template class Protocol<WavegenNSVirtual>;
