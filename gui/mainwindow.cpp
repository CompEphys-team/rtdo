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
#include "config.h"
#include "util.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    channel_setup(new ChannelSetupDialog),
    vclamp_setup(new VClampSetupDialog),
    wavegen_setup(new WavegenSetupDialog),
    model_setup(new ModelSetupDialog),
    performance(new PerformanceDialog),
    compiler(new CompileRunner),
    module(nullptr),
    wavegen(new Runner(XMLModel::WaveGen)),
    wavegenNS(new Runner(XMLModel::WaveGenNoveltySearch))
{
    ui->setupUi(this);
    connect(channel_setup, SIGNAL(channelsUpdated()), vclamp_setup, SIGNAL(channelsUpdated()));
    connect(ui->actionVoltage_clamp, SIGNAL(triggered()), vclamp_setup, SLOT(open()));
    connect(ui->actionChannel_setup, SIGNAL(triggered()), channel_setup, SLOT(open()));
    connect(ui->actionWavegen_setup, SIGNAL(triggered()), wavegen_setup, SLOT(open()));
    connect(ui->actionModel_setup, SIGNAL(triggered()), model_setup, SLOT(open()));
    connect(ui->actionPerformance, SIGNAL(triggered()), performance, SLOT(open()));
    connect(wavegen, SIGNAL(processCompleted(bool)), this, SLOT(wavegenComplete(bool)));
    connect(wavegenNS, SIGNAL(processCompleted(bool)), this, SLOT(wavegenNSComplete(bool)));
    connect(ui->wavegen_stop, SIGNAL(clicked()), wavegen, SLOT(stop()));
    connect(ui->wavegen_stop_NS, SIGNAL(clicked()), wavegenNS, SLOT(stop()));

#ifndef CONFIG_RT
    ui->actionChannel_setup->setEnabled(false);
#endif
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_vclamp_start_clicked()
{
    if ( !module ) {
        try {
            module = new Module();
        } catch (runtime_error &e) {
            cerr << "Failed to load module: " << e.what() << endl;
            return;
        }
        connect(module, SIGNAL(complete(int)), this, SLOT(vclampComplete(int)));
        connect(ui->vclamp_stop, SIGNAL(clicked()), module, SLOT(stop()));
    }
    module->push( [=](int) {
        module->vclamp->run();
    });
    ui->vclamp_compile->setEnabled(false);
    ui->vclamp_start->setEnabled(false);
    ui->vclamp_stop->setEnabled(true);
}

void MainWindow::vclampComplete(int handle)
{
    ui->vclamp_compile->setEnabled(true);
    ui->vclamp_start->setEnabled(true);
    ui->vclamp_stop->setEnabled(false);
}

void MainWindow::on_vclamp_compile_clicked()
{
    QApplication::setOverrideCursor(QCursor(Qt::BusyCursor));
    if ( module ) {
        delete module;
        module = nullptr;
    }
    ui->vclamp_compile->setEnabled(false);
    ui->vclamp_start->setEnabled(false);
    compiler->setType(XMLModel::VClamp);
    if ( compiler->start() )
        ui->vclamp_start->setEnabled(true);
    ui->vclamp_compile->setEnabled(true);
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

void MainWindow::on_wavegen_start_clicked()
{
    if ( !wavegen->start() )
        return;
    ui->wavegen_compile->setEnabled(false);
    ui->wavegen_start->setEnabled(false);
    ui->wavegen_stop->setEnabled(true);
}

void MainWindow::wavegenComplete(bool successfully)
{
    ui->wavegen_compile->setEnabled(true);
    ui->wavegen_start->setEnabled(true);
    ui->wavegen_stop->setEnabled(false);
}

void MainWindow::on_wavegen_compile_clicked()
{
    QApplication::setOverrideCursor(QCursor(Qt::BusyCursor));
    ui->wavegen_compile->setEnabled(false);
    ui->wavegen_start->setEnabled(false);
    compiler->setType(XMLModel::WaveGen);
    if ( compiler->start() )
        ui->wavegen_start->setEnabled(true);
    ui->wavegen_compile->setEnabled(true);
    QApplication::restoreOverrideCursor();
}

void MainWindow::on_wavegen_start_NS_clicked()
{
    if ( !wavegenNS->start() )
        return;
    ui->wavegen_compile_NS->setEnabled(false);
    ui->wavegen_start_NS->setEnabled(false);
    ui->wavegen_stop_NS->setEnabled(true);
}

void MainWindow::wavegenNSComplete(bool successfully)
{
    ui->wavegen_compile_NS->setEnabled(true);
    ui->wavegen_start_NS->setEnabled(true);
    ui->wavegen_stop_NS->setEnabled(false);
}

void MainWindow::on_wavegen_compile_NS_clicked()
{
    QApplication::setOverrideCursor(QCursor(Qt::BusyCursor));
    ui->wavegen_compile_NS->setEnabled(false);
    ui->wavegen_start_NS->setEnabled(false);
    compiler->setType(XMLModel::WaveGenNoveltySearch);
    if ( compiler->start() )
        ui->wavegen_start_NS->setEnabled(true);
    ui->wavegen_compile_NS->setEnabled(true);
    QApplication::restoreOverrideCursor();
}
