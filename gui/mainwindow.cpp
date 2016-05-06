/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2015-12-03

--------------------------------------------------------------------------*/
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "run.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QInputDialog>
#include <QTextStream>
#include <QCheckBox>
#include "config.h"
#include "util.h"
#include <fstream>
#include "fixparamdialog.h"
#include <unistd.h>
#include <sys/stat.h>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    channel_setup(new ChannelSetupDialog(this)),
    vclamp_setup(new VClampSetupDialog(this)),
    wavegen_setup(new WavegenSetupDialog(this)),
    model_setup(new ModelSetupDialog(this)),
    performance(new PerformanceDialog(this)),
    compiler(new CompileRunner),
    wavegen(new Runner(XMLModel::WaveGen)),
    wgmodule(nullptr),
    module(nullptr),
    protocol(nullptr),
    offlineNoAsk(false)
{
    ui->setupUi(this);
    ui->stackedWidget->setCurrentWidget(ui->pSetup);
    ui->menuActions->setEnabled(false);
    ui->menuOffline->setEnabled(false);

    connect(&*channel_setup, SIGNAL(channelsUpdated()), &*vclamp_setup, SIGNAL(channelsUpdated()));
    connect(ui->actionVoltage_clamp, SIGNAL(triggered()), &*vclamp_setup, SLOT(open()));
    connect(ui->actionChannel_setup, SIGNAL(triggered()), &*channel_setup, SLOT(open()));
    connect(ui->actionWavegen_setup, SIGNAL(triggered()), &*wavegen_setup, SLOT(open()));
    connect(ui->actionModel_setup, SIGNAL(triggered()), &*model_setup, SLOT(open()));
    connect(ui->actionPerformance, SIGNAL(triggered()), &*performance, SLOT(open()));

    connect(ui->vclamp_compile, SIGNAL(clicked(bool)), this, SLOT(compile()));
    connect(ui->wavegen_compile, SIGNAL(clicked(bool)), this, SLOT(compile()));
    connect(ui->wavegen_compile_NS, SIGNAL(clicked(bool)), this, SLOT(compile()));

    connect(ui->wavegen_stop, SIGNAL(clicked()), &*wavegen, SLOT(stop()));
    connect(&*wavegen, SIGNAL(processCompleted(bool)), this, SLOT(wavegenComplete()));

    connect(ui->menuActions, SIGNAL(triggered(QAction*)), this, SLOT(qAction(QAction*)));
    connect(ui->menuOffline, SIGNAL(triggered(QAction*)), this, SLOT(offlineAction(QAction*)));
    connect(ui->VCApply, SIGNAL(clicked(bool)), &*vclamp_setup, SLOT(open()));
    connect(ui->btnZeroOutputs, SIGNAL(clicked(bool)), this, SLOT(zeroOutputs()));

    connect(ui->notesArea, &QPlainTextEdit::modificationChanged, [=](bool y){
        ui->btnNotesSave->setEnabled(y && !module->outdir.empty());
    });

    // Todo: Config should probably emit its own signal.
    connect(&*vclamp_setup, SIGNAL(configChanged()), this, SLOT(updateConfigFields()));
    connect(this, SIGNAL(configChanged()), this, SLOT(updateConfigFields()));

#ifndef CONFIG_RT
    ui->actionChannel_setup->setEnabled(false);
#endif

    const char *home = getenv("HOME");
    QFile recentCfgFile(QString(home)+"/.rtdo/recentConfig");
    if ( recentCfgFile.open(QIODevice::ReadOnly | QIODevice::Text) ) {
        QTextStream s(&recentCfgFile);
        QString line;
        while ( s.readLineInto(&line) ) {
            if ( !access(line.toStdString().c_str(), R_OK) ) {
                recentConfigs << line;
            }
        }
        if ( recentConfigs.size() ) {
            config = new conf::Config(recentConfigs.first().toStdString());
            emit configChanged();
            updateRecent(&recentConfigs);
        }
    }
    QFile recentProtFile(QString(home)+"/.rtdo/recentProtocols");
    if ( recentProtFile.open(QIODevice::ReadOnly | QIODevice::Text) ) {
        QTextStream s(&recentProtFile);
        QString line;
        while ( s.readLineInto(&line) ) {
            if ( !access(line.toStdString().c_str(), R_OK) ) {
                recentProtocols << line;
            }
        }
        if ( recentProtocols.size() )
            updateRecent(&recentProtocols);
    }
}

MainWindow::~MainWindow()
{
    const char *home = getenv("HOME");
    if ( access((QString(home)+"/.rtdo").toLocal8Bit().constData(), F_OK) ) {
        mkdir((QString(home)+"/.rtdo").toLocal8Bit().constData(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
    QFile recentCfgFile(QString(home)+"/.rtdo/recentConfig");
    if ( recentCfgFile.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text) ) {
        QTextStream s(&recentCfgFile);
        for ( QString str : recentConfigs ) {
            s << str << endl;
        }
    }
    QFile recentProtFile(QString(home)+"/.rtdo/recentProtocols");
    if ( recentProtFile.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text) ) {
        QTextStream s(&recentProtFile);
        for ( QString str : recentProtocols ) {
            s << str << endl;
        }
    }

    delete wgmodule;
    delete module;
    delete ui;
}

void MainWindow::updateConfigFields()
{
    ui->VCGain->setValue(config->vc.gain);
    ui->VCResistance->setValue(config->vc.resistance);
}


// -------------------------------------------- pSetup -----------------------------------------------------------------
void MainWindow::compile()
{
    QApplication::setOverrideCursor(QCursor(Qt::BusyCursor));
    ui->pSetup->setEnabled(false);
    QObject *snd = QObject::sender();
    if ( snd == ui->vclamp_compile )
        compiler->setType(XMLModel::VClamp);
    else if ( snd == ui->wavegen_compile )
        compiler->setType(XMLModel::WaveGen);
    else if ( snd == ui->wavegen_compile_NS )
        compiler->setType(XMLModel::WaveGenNoveltySearch);
    compiler->start();
    ui->pSetup->setEnabled(true);
    QApplication::restoreOverrideCursor();
}

void MainWindow::on_actionSave_configuration_triggered()
{
    QString file = QFileDialog::getSaveFileName(this, QString("Select configuration file..."), QString(), QString("*.xml"));
    if ( file.isEmpty() )
        return;
    if ( !file.endsWith(".xml") )
        file.append(".xml");
    config->save(file.toStdString());

    updateRecent(&recentConfigs, file);
}

void MainWindow::on_actionLoad_configuration_triggered()
{
    QString file = QFileDialog::getOpenFileName(this, QString("Select configuration file..."), QString(), QString("*.xml"));
    if ( file.isEmpty() )
        return;
    delete config;
    config = new conf::Config(file.toStdString());
    emit configChanged();

    updateRecent(&recentConfigs, file);
}

void MainWindow::loadRecentConfiguration()
{
    QAction *action = qobject_cast<QAction *>(sender());
    QString file = action->data().toString();
    config = new conf::Config(file.toStdString());
    emit configChanged();
    updateRecent(&recentConfigs, file);
}


// -------------------------- pWavegen ---------------------------------------------------------------------------------
void MainWindow::on_wavegen_start_clicked()
{
    if ( !wavegen->start() )
        return;
    ui->wavegen_start->setEnabled(false);
    ui->wavegen_start_NS->setEnabled(false);
    ui->pWavegen2Setup->setEnabled(false);
    ui->wavegen_stop->setEnabled(true);
}

void MainWindow::on_wavegen_start_NS_clicked()
{
    try {
        wgmodule = new Module<WavegenNSVirtual>(this);
    } catch (runtime_error &e ) {
        cerr << e.what() << endl;
        wgmodule = nullptr;
        return;
    }
    connect(wgmodule, SIGNAL(complete(int)), this, SLOT(wavegenComplete()));
    connect(ui->wavegen_stop, SIGNAL(clicked(bool)), wgmodule, SLOT(stop()));

    ui->wavegen_start->setEnabled(false);
    ui->wavegen_start_NS->setEnabled(false);
    ui->pWavegen2Setup->setEnabled(false);
    ui->wavegen_stop->setEnabled(true);

    wgmodule->push("runAll", [=](int){
        std::ofstream wf(wgmodule->outdir + "/wave.stim");
        std::ofstream cf(wgmodule->outdir + "/currents.log");
        wgmodule->obj->runAll(wf, cf);
    });
    wgmodule->start();
}

void MainWindow::wavegenComplete()
{
    delete wgmodule;
    wgmodule = nullptr;

    ui->wavegen_start->setEnabled(true);
    ui->wavegen_start_NS->setEnabled(true);
    ui->pWavegen2Setup->setEnabled(true);
    ui->wavegen_stop->setEnabled(false);
}


// ------------------------------------ pExperiment -----------------------------------------------------------------------
//******** Actions ***********
void MainWindow::qAction(QAction *action)
{
    if ( action == ui->actVCFrontload ) {
        QMessageBox::StandardButton reply;
        reply = QMessageBox::question(this, action->text(), "Fit models during this action?",
                                      QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel);
        if ( reply == QMessageBox::Cancel )
            return;
        protocol->appendItem(ActionListModel::VCFrontload, reply == QMessageBox::Yes);
    } else if ( action == ui->actVCCycle ) {
        QMessageBox::StandardButton reply;
        reply = QMessageBox::question(this, action->text(), "Fit models during this action?",
                                      QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel);
        if ( reply == QMessageBox::Cancel )
            return;
        protocol->appendItem(ActionListModel::VCCycle, reply == QMessageBox::Yes);
    } else if ( action == ui->actVCRun ) {
        bool ok;
        int nEpochs = QInputDialog::getInt(this, action->text(), "Number of epochs (0 = unlimited)", 0, 0, INT_MAX, 1, &ok);
        if ( !ok )
            return;
        protocol->appendItem(ActionListModel::VCRun, nEpochs);
    } else if ( action == ui->actModelsSaveAll ) {
        protocol->appendItem(ActionListModel::ModelsSaveAll);
    } else if ( action == ui->actModelsSaveEval ) {
        protocol->appendItem(ActionListModel::ModelsSaveEval);
    } else if ( action == ui->actModelStimulate ) {
        protocol->appendItem(ActionListModel::ModelStimulate);
    } else if ( action == ui->actTracesSave ) {
        protocol->appendItem(ActionListModel::TracesSave);
    } else if ( action == ui->actTracesDrop ) {
        protocol->appendItem(ActionListModel::TracesDrop);
    } else if ( action == ui->actParamFix ) {
        FixParamDialog fpd(this);
        if ( fpd.exec() != QDialog::Accepted || fpd.param < 0 )
            return;
        protocol->appendItem(ActionListModel::ParamFix, fpd.param, fpd.value);
    } else if ( action == ui->actionSave_protocol ) {
        QString file = QFileDialog::getSaveFileName(this, QString("Select protocol file..."), QString(), QString("*.xml"));
        if ( !file.isEmpty() ) {
            if ( !file.endsWith(".xml") )
                file.append(".xml");
            if ( !protocol->save(file.toStdString()) )
                cerr << "Failed to write protocol to " << file.toStdString() << endl;
            else
                updateRecent(&recentProtocols, file);
        }
    } else { // Load protocol (recent or "...")
        QString file;
        if ( action == ui->actionLoad_protocol ) {
            file = QFileDialog::getOpenFileName(this, QString("Select protocol file..."), QString(), QString("*.xml"));
        } else {
            file = action->data().toString();
        }
        if ( !file.isEmpty() ) {
            if ( !protocol->load(file.toStdString()) )
                cerr << "Failed to open protocol from " << file.toStdString() << endl;
            else
                updateRecent(&recentProtocols, file);
        }
    }
}

void MainWindow::actionComplete(int handle)
{
    if ( !module->busy() ) {
        ui->btnQStart->setText("Start");
        ui->btnQSkip->setEnabled(false);
        ui->pExperiment2Setup->setEnabled(true);
        ui->pExperimentReset->setEnabled(true);
    }
}

void MainWindow::on_btnQRemove_clicked()
{
    protocol->removeItem(ui->actionQ->currentIndex());
}

void MainWindow::on_btnQStart_clicked()
{
    if ( module->busy() ) {
        module->stop();
        protocol->clear();
    } else {
        if ( module->qSize() ) {
            module->start();
            ui->btnQStart->setText("Stop");
            ui->btnQSkip->setEnabled(true);
            ui->pExperiment2Setup->setEnabled(false);
            ui->pExperimentReset->setEnabled(false);
        }
    }
}

void MainWindow::on_btnQSkip_clicked()
{
    protocol->removeItem(ui->actionQ->indexAt(QPoint(0,0)));
}

// ********* Voltage clamp config **************
void MainWindow::on_VCApply_clicked()
{
    config->vc.gain = ui->VCGain->value();
    config->vc.resistance = ui->VCResistance->value();
    emit configChanged();
}

void MainWindow::zeroOutputs()
{
    for ( const Channel &c : config->io.channels ) {
        if ( c.direction() == Channel::AnalogOut ) {
            if ( c.offsetSource() )
                c.zero();
            else
                c.write(0.0);
        }
    }
}

// *************** Notes **********************
void MainWindow::on_btnNotesLoad_clicked()
{
    QString filename = QFileDialog::getOpenFileName(this, QString("Select note template..."),
                                                    QString::fromStdString(config->output.dir), QString("*.txt"));
    if ( filename.isEmpty() )
        return;
    QFile file(filename);
    if ( !file.open(QFile::ReadOnly | QFile::Text) ) {
        cerr << "Failed to read from file " << filename.toStdString() << endl;
        return;
    }
    ui->notesArea->document()->setPlainText(file.readAll());
}

void MainWindow::on_btnNotesSave_clicked()
{
    QString filename = QString::fromStdString(module->outdir + "/notes.txt");
    QFile file(filename);
    if ( !file.open(QFile::WriteOnly | QFile::Text) ) {
        cerr << "Failed to write to file " << filename.toStdString() << endl;
        return;
    }
    QTextStream out(&file);
    out << ui->notesArea->document()->toPlainText();
    ui->notesArea->document()->setModified(false);
    cout << "Saved notes to " << filename.toStdString() << endl;
}

// ******************** Offline ******************************
void MainWindow::offlineAction(QAction *action)
{
    if ( !offlineNoAsk ) {
        QMessageBox mb(QMessageBox::Warning,
                       "Offline action",
                       "Offline actions interferes with live progress. Continue and discard unsaved progress?",
                       QMessageBox::Yes | QMessageBox::No);
        QCheckBox *cb = new QCheckBox("Don't ask again");
        mb.setCheckBox(cb);
        mb.exec();
        if ( mb.result() == QMessageBox::No )
            return;

        if ( cb->isChecked() )
            offlineNoAsk = true;
    }

    if ( action == ui->offline_stimulateBest ) {
        string filename = QFileDialog::getOpenFileName(this, QString("Select model dump..."),
                                                        QString(), QString("*_modelsAll.log *_modelsEval.log")).toStdString();
        if ( filename.empty() )
            return;

        config = new conf::Config(filename.substr(0, filename.find_last_of('_')) + "_config.xml");
        emit configChanged();

        delete protocol;
        delete module;
        protocol = nullptr;
        module = nullptr;
        if ( !pExpInit() )
            pExp2Setup();

        ui->outdirDisplay->setText(QString("Loaded ") + QString::fromStdString(filename));

        ifstream file(filename);
        vector<double> params = read_model_dump(file, 1);

        module->obj->injectModel(params, 0);

        ofstream tf(filename + ".winner.simtrace");
        tf << "# Traces from best model, see top-ranked model in parent file" << endl;

        vector<vector<double>> traces = module->obj->stimulateModel(0);
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
    } else if ( action == ui->offline_loadTraces ) {
        string filename = QFileDialog::getOpenFileName(this, QString("Select trace file..."),
                                                        QString(), QString("*.traces")).toStdString();
        if ( filename.empty() )
            return;

        config = new conf::Config(filename.substr(0, filename.find_last_of('.')) + "_config.xml");
        emit configChanged();

        delete protocol;
        delete module;
        protocol = nullptr;
        module = nullptr;
        if ( !pExpInit() )
            pExp2Setup();

        ui->outdirDisplay->setText(QString("Loaded ") + QString::fromStdString(filename));

        QFile file(dirname(filename) + "/notes.txt");
        if ( file.open(QFile::ReadOnly | QFile::Text) ) {
            ui->notesArea->document()->setPlainText(file.readAll());
        }

        ifstream tf(filename);
        module->obj->data()->load(tf);

        QMessageBox::StandardButton reply;
        reply = QMessageBox::question(this, "Append outputs", "Add outputs to existing directory?",
                                      QMessageBox::Yes | QMessageBox::No);
        string saveLoc("saved to a new directory");
        if ( reply == QMessageBox::Yes ) {
            if ( module->append(dirname(filename).toStdString()) )
                saveLoc = "appended to the existing directory";
            else
                cerr << "Append failed." << endl;
        }
        cout << "Loading complete. You may now proceed as usual; output will be " << saveLoc << "." << endl;
    } else if ( action == ui->offline_stimulateBestCC ) {
        string filename = QFileDialog::getOpenFileName(this, QString("Select model dump..."),
                                                        QString(), QString("*_modelsAll.log *_modelsEval.log *.simtrace")).toStdString();
        if ( filename.empty() )
            return;

        string stimfilename = QFileDialog::getOpenFileName(this, QString("Select current clamp stimulation..."),
                                                           QString(), QString("*.stim")).toStdString();
        if ( stimfilename.empty() )
            return;

        if ( !filename.compare(filename.find_last_of('.'), string::npos, ".simtrace") )
            config = new conf::Config(filename.substr(0, filename.find_last_of('.')) + "_config.xml");
        else
            config = new conf::Config(filename.substr(0, filename.find_last_of('_')) + "_config.xml");
        emit configChanged();

        delete protocol;
        delete module;
        protocol = nullptr;
        module = nullptr;
        if ( !pExpInit() )
            pExp2Setup();

        ui->outdirDisplay->setText(QString("Loaded ") + QString::fromStdString(filename));

        ifstream file(filename);
        vector<double> params = read_model_dump(file, 1);
        file.close();

        module->obj->injectModel(params, 0);

        file.open(stimfilename);
        char buffer[1024];
        while (((file.peek() == '%') || (file.peek() == '\n') || (file.peek() == ' ') || (file.peek() == '#')) && file.good()) { // remove comments
            file.getline( buffer, 1024 );
        }

        inputSpec wave;
        file >> wave;

        vector<double> trace = module->obj->getCCVoltageTrace(wave, 0);

        filename += ".CC.simtrace";
        ofstream tf(filename);
        tf << "# Traces from best model, see top-ranked model in parent file" << endl;
        tf << endl << endl << "Time\tInput_current\tMembrane_potential" << endl;
        double t = 0.0;
        double I = wave.baseV;
        int sn = 0;
        for ( double V : trace ) {
            tf << t << '\t' << I << '\t' << V << endl;
            t += config->io.dt;
            if ( sn < wave.N && t > wave.st.at(sn) ) {
                I = wave.V.at(sn++);
            }
        }

        ui->outdirDisplay->setText(QString::fromStdString(filename));
    }
}

// ************** Misc ****************************
bool MainWindow::pExpInit()
{
    try {
        module = new Module<Experiment>(this);
        protocol = new ActionListModel(module);
    } catch (runtime_error &e ) {
        cerr << e.what() << endl;
        module = nullptr;
        return false;
    }
    connect(module, SIGNAL(complete(int)), this, SLOT(actionComplete(int)));
    connect(module, SIGNAL(outdirSet()), this, SLOT(outdirSet()));
    zeroOutputs();
    ui->outdirDisplay->clear();
    ui->btnNotesSave->setEnabled(false);
    ui->actionQ->setModel(protocol);
    return true;
}

void MainWindow::on_pExperimentReset_clicked()
{
    QMessageBox::StandardButton reply;
    reply = QMessageBox::question(this, "Reset", "Reset experiment mode and discard unsaved progress?",
                                  QMessageBox::Yes | QMessageBox::No);
    if ( reply == QMessageBox::Yes ) {
        QApplication::setOverrideCursor(QCursor(Qt::BusyCursor));
        delete protocol;
        delete module;
        protocol = nullptr;
        module = nullptr;
        if ( !pExpInit() )
            pExp2Setup();
    }
    QApplication::restoreOverrideCursor();
}

void MainWindow::outdirSet()
{
    ui->outdirDisplay->setText(QString::fromStdString(module->outdir));
    ui->btnNotesSave->setEnabled(!ui->notesArea->document()->isEmpty());
}


// -------------------------------------------- page transitions ------------------------------------------------------------
void MainWindow::on_pSetup2Experiment_clicked()
{
    QApplication::setOverrideCursor(QCursor(Qt::BusyCursor));
    if ( pExpInit() ) {
        ui->menuActions->setEnabled(true);
        ui->menuOffline->setEnabled(true);
        for ( QAction *a : ui->menuConfig->actions() ) {
            if ( a != ui->actionVoltage_clamp && a != ui->actionPerformance )
                a->setEnabled(false);
        }
        vclamp_setup->setExperimentMode(true);
        ui->stackedWidget->setCurrentWidget(ui->pExperiment);
    }
    QApplication::restoreOverrideCursor();
}

void MainWindow::on_pSetup2Wavegen_clicked()
{
    ui->stackedWidget->setCurrentWidget(ui->pWavegen);
}

void MainWindow::on_pWavegen2Setup_clicked()
{
    ui->stackedWidget->setCurrentWidget(ui->pSetup);
}

void MainWindow::on_pExperiment2Setup_clicked()
{
    QMessageBox::StandardButton reply;
    reply = QMessageBox::question(this, "Exit", "Exit experiment mode and discard unsaved progress?",
                                  QMessageBox::Yes | QMessageBox::No);
    if ( reply == QMessageBox::Yes ) {
        QApplication::setOverrideCursor(QCursor(Qt::BusyCursor));
        delete protocol;
        delete module;
        protocol = nullptr;
        module = nullptr;
        pExp2Setup();
        QApplication::restoreOverrideCursor();
    }
}

void MainWindow::pExp2Setup()
{
    ui->menuActions->setEnabled(false);
    ui->menuOffline->setEnabled(false);
    for ( QAction *a : ui->menuConfig->actions() ) {
        if ( a != ui->actionVoltage_clamp && a != ui->actionPerformance )
            a->setEnabled(true);
    }
    vclamp_setup->setExperimentMode(false);
    ui->stackedWidget->setCurrentWidget(ui->pSetup);
}


// -------------------------------------------- Miscellaneous ------------------------------------------------------------
void MainWindow::updateRecent(QStringList *list, QString entry)
{
    if ( entry.size() )
        list->push_front(entry);
    list->removeDuplicates();
    while ( list->size() > 5 )
        list->pop_back();

    QMenu *m;
    QAction *head = nullptr;
    QAction *tail = nullptr;
    if ( list == &recentConfigs ) {
        m = ui->menuConfig;
        head = ui->actionLoad_configuration;
    } else if ( list == &recentProtocols ) {
        m = ui->menuActions;
        head = ui->actionLoad_protocol;
    } else {
        return;
    }

    bool skip = (head != nullptr);
    for ( QAction *a : m->actions() ) {
        if ( skip && a != head )
            continue;
        if ( a == head ) {
            skip = false;
            continue;
        }
        if ( a == tail )
            break;
        m->removeAction(a);
        skip = false;
    }

    int i = 0;
    for ( QString s : *list ) {
        QAction *a = new QAction(QString("&%1 %2").arg(++i).arg(basename_nosuffix(s)), this);
        a->setData(s);
        a->setToolTip(s);
        m->insertAction(tail, a);

        if ( m == ui->menuConfig )
            connect(a, SIGNAL(triggered(bool)), this, SLOT(loadRecentConfiguration()));
    }
}
