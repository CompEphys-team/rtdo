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
#include "globals.h"
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
    model_setup(new ModelSetupDialog)
{
    ui->setupUi(this);
    connect(channel_setup, SIGNAL(channelsUpdated()), vclamp_setup, SIGNAL(channelsUpdated()));
    connect(ui->actionVoltage_clamp, SIGNAL(triggered()), vclamp_setup, SLOT(open()));
    connect(ui->actionChannel_setup, SIGNAL(triggered()), channel_setup, SLOT(open()));
    connect(ui->actionWavegen_setup, SIGNAL(triggered()), wavegen_setup, SLOT(open()));
    connect(ui->actionModel_setup, SIGNAL(triggered()), model_setup, SLOT(open()));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_vclamp_start_clicked()
{
    run_vclamp_start();
}

void MainWindow::on_vclamp_stop_clicked()
{
    run_vclamp_stop();
}

void MainWindow::on_vclamp_compile_clicked()
{
    compile_model(XMLModel::VClamp);
}

void MainWindow::on_actionSave_configuration_triggered()
{
    QString file = QFileDialog::getSaveFileName(0, QString("Select configuration file..."), QString(), QString("*.xml"));
    if ( file.isEmpty() )
        return;
    if ( !file.endsWith(".xml") )
        file.append(".xml");
    config.save(file.toStdString());
}

void MainWindow::on_actionLoad_configuration_triggered()
{
    QString file = QFileDialog::getOpenFileName(0, QString("Select configuration file..."), QString(), QString("*.xml"));
    if ( file.isEmpty() )
        return;
    config.load(file.toStdString());
}

void MainWindow::on_wavegen_start_clicked()
{
    run_wavegen_start();
}

void MainWindow::on_wavegen_stop_clicked()
{
    run_wavegen_stop();
}

void MainWindow::on_wavegen_compile_clicked()
{
    compile_model(XMLModel::WaveGen);
}
