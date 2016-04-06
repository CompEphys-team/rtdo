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
#include <fstream>
#include "config.h"
#include "util.h"
#include "backlog.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    channel_setup(new ChannelSetupDialog),
    vclamp_setup(new VClampSetupDialog),
    wavegen_setup(new WavegenSetupDialog),
    model_setup(new ModelSetupDialog),
    performance(new PerformanceDialog),
    compiler(new CompileRunner),
    wavegen(new Runner(XMLModel::WaveGen)),
    wavegenNS(new Runner(XMLModel::WaveGenNoveltySearch)),
    module(nullptr)
{
    ui->setupUi(this);
    ui->stackedWidget->setCurrentWidget(ui->pSetup);

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

#ifndef CONFIG_RT
    ui->actionChannel_setup->setEnabled(false);
#endif
}

MainWindow::~MainWindow()
{
    delete module;
    delete ui;
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
    QString file = QFileDialog::getSaveFileName(0, QString("Select configuration file..."), QString(), QString("*.xml"));
    if ( file.isEmpty() )
        return;
    if ( !file.endsWith(".xml") )
        file.append(".xml");
    config->save(file.toStdString());
}

void MainWindow::on_actionLoad_configuration_triggered()
{
    QString file = QFileDialog::getOpenFileName(0, QString("Select configuration file..."), QString(), QString("*.xml"));
    if ( file.isEmpty() )
        return;
    delete config;
    config = new conf::Config(file.toStdString());
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
void MainWindow::qAction(QAction *action)
{
    int handle = -1;
    if ( action == ui->actVCFrontload ) {
        handle = module->push( [=](int) {
            for ( unsigned int i = 0; i < config->vc.cacheSize && !module->vclamp->stopFlag; i++ )
                module->vclamp->cycle(false);
        });
    } else if ( action == ui->actVCCycle ) {
        handle = module->push( [=](int) {
            module->vclamp->cycle(false);
        });
    } else if ( action == ui->actVCRun ) {
        handle = module->push( [=](int) {
            module->vclamp->run();
        });
    } else if ( action == ui->actModelsSaveAll ) {
        handle = module->push( [=](int h) {
            ofstream logf(module->outdir + "/" + to_string(h) + "_modelsAll.log");
            module->vclamp->log()->score();
            write_backlog(logf, module->vclamp->log()->sort(backlog::BacklogVirtual::ErrScore, true), false);
        });
    } else if ( action == ui->actModelsSaveEval ) {
        handle = module->push( [=](int h) {
            ofstream logf(module->outdir + "/" + to_string(h) + "_modelsEval.log");
            module->vclamp->log()->score();
            write_backlog(logf, module->vclamp->log()->sort(backlog::BacklogVirtual::ErrScore, true), true);
        });
    } else if ( action == ui->actTracesSave ) {
        handle = module->push( [=](int h) {
            ofstream tf(module->outdir + "/" + to_string(h) + ".traces");
            module->vclamp->data()->dump(tf);
        });
    } else if ( action == ui->actTracesDrop ) {
        handle = module->push( [=](int) {
            module->vclamp->data()->clear();
        });
    }

    QString label = QString("(%1) ").arg(handle);
    QWidget *widget = action->associatedWidgets().first();
    if ( widget ) {
        QMenu *menu = dynamic_cast<QMenu*>(widget);
        if ( menu )
            label += menu->title() + QString(": ");
    }
    label += action->text();
    QListWidgetItem *item = new QListWidgetItem(label, ui->actionQ);
    item->setData(Qt::UserRole, QVariant(handle));
}

void MainWindow::actionComplete(int handle)
{
    QListWidgetItem *item = ui->actionQ->item(0);
    while ( item && item->data(Qt::UserRole) <= handle ) {
        delete item;
        item = ui->actionQ->item(0);
    }

    if ( !module->busy() ) {
        ui->btnQStart->setText("Start");
        ui->btnQSkip->setEnabled(false);
    }
}

void MainWindow::on_btnQRemove_clicked()
{
    if ( ui->actionQ->currentIndex().row() > 0 || !module->busy() ) {
        QListWidgetItem *item = ui->actionQ->currentItem();
        if ( module->erase(item->data(Qt::UserRole).toInt()) )
            delete item;
    } else if ( ui->actionQ->currentIndex().row() == 0 ) {
        on_btnQSkip_clicked();
    }
}

void MainWindow::on_btnQStart_clicked()
{
    if ( module->busy() ) {
        module->stop();
        ui->actionQ->clear();
    } else {
        if ( module->qSize() ) {
            module->start();
            ui->btnQStart->setText("Stop");
            ui->btnQSkip->setEnabled(true);
        }
    }
}

void MainWindow::on_btnQSkip_clicked()
{
    if ( module->busy() ) {
        module->skip();
    }
}


// -------------------------------------------- page transitions ------------------------------------------------------------
void MainWindow::on_pSetup2Experiment_clicked()
{
    QApplication::setOverrideCursor(QCursor(Qt::BusyCursor));
    try {
        module = new Module(this);
        connect(module, SIGNAL(complete(int)), this, SLOT(actionComplete(int)));
        ui->menuActions->setEnabled(true);
        ui->stackedWidget->setCurrentWidget(ui->pExperiment);
    } catch ( runtime_error &e ) {
        cerr << e.what() << endl;
        module = nullptr;
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
        delete module;
        module = nullptr;
        QApplication::restoreOverrideCursor();
        ui->menuActions->setEnabled(false);
        ui->stackedWidget->setCurrentWidget(ui->pSetup);
    }
}
