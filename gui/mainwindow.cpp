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
    wavegenNS(new Runner(XMLModel::WaveGenNoveltySearch)),
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
    connect(ui->wavegen_stop, SIGNAL(clicked()), &*wavegenNS, SLOT(stop()));
    connect(&*wavegen, SIGNAL(processCompleted(bool)), this, SLOT(wavegenComplete(bool)));
    connect(&*wavegenNS, SIGNAL(processCompleted(bool)), this, SLOT(wavegenComplete(bool)));

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
}

MainWindow::~MainWindow()
{
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
}

void MainWindow::on_actionLoad_configuration_triggered()
{
    QString file = QFileDialog::getOpenFileName(this, QString("Select configuration file..."), QString(), QString("*.xml"));
    if ( file.isEmpty() )
        return;
    delete config;
    config = new conf::Config(file.toStdString());
    emit configChanged();
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
    if ( !wavegenNS->start() )
        return;
    ui->wavegen_start->setEnabled(false);
    ui->wavegen_start_NS->setEnabled(false);
    ui->pWavegen2Setup->setEnabled(false);
    ui->wavegen_stop->setEnabled(true);
}

void MainWindow::wavegenComplete(bool successfully)
{
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
    } else if ( action == ui->actionSave_protocol ) {
        QString file = QFileDialog::getSaveFileName(this, QString("Select protocol file..."), QString(), QString("*.xml"));
        if ( !file.isEmpty() ) {
            if ( !file.endsWith(".xml") )
                file.append(".xml");
            if ( !protocol->save(file.toStdString()) )
                cerr << "Failed to write protocol to " << file.toStdString() << endl;
        }
    } else if ( action == ui->actionLoad_protocol ) {
        QString file = QFileDialog::getOpenFileName(this, QString("Select protocol file..."), QString(), QString("*.xml"));
        if ( !file.isEmpty() ) {
            if ( !protocol->load(file.toStdString()) )
                cerr << "Failed to open protocol from " << file.toStdString() << endl;
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

        delete module;
        module = new Module(this);
        ui->outdirDisplay->setText(dirname(filename));

        ifstream file(filename);
        vector<double> params = read_model_dump(file, 1);

        module->vclamp->injectModel(params, 0);

        ofstream tf(filename + ".winner.simtrace");
        tf << "# Traces from best model, see top-ranked model in parent file" << endl;

        vector<vector<double>> traces = module->vclamp->stimulateModel(0);
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
    }
}

// ************** Misc ****************************
bool MainWindow::pExpInit()
{
    try {
        module = new Module(this);
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
        ui->actionChannel_setup->setEnabled(false);
        ui->actionModel_setup->setEnabled(false);
        ui->actionWavegen_setup->setEnabled(false);
        ui->actionLoad_configuration->setEnabled(false);
        ui->actionSave_configuration->setEnabled(false);
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
    ui->actionChannel_setup->setEnabled(true);
    ui->actionModel_setup->setEnabled(true);
    ui->actionWavegen_setup->setEnabled(true);
    ui->actionLoad_configuration->setEnabled(true);
    ui->actionSave_configuration->setEnabled(true);
    vclamp_setup->setExperimentMode(false);
    ui->stackedWidget->setCurrentWidget(ui->pSetup);
}
