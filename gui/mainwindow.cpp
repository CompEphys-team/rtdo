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
    vclamp_setup(new VClampSetupDialog)
{
    ui->setupUi(this);
    on_simparams_reset_clicked();
    connect(channel_setup, SIGNAL(channelsUpdated()), vclamp_setup, SIGNAL(channelsUpdated()));
    connect(ui->actionVoltage_clamp, SIGNAL(triggered()), vclamp_setup, SLOT(open()));
    connect(ui->actionChannel_setup, SIGNAL(triggered()), channel_setup, SLOT(open()));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_outdir_browse_clicked()
{
    QString file, dir;
    dir = dirname(ui->outdir->text().toStdString());
    file = QFileDialog::getExistingDirectory(this, QString("Select data output directory..."), dir);
    if ( !file.isEmpty() )
        ui->outdir->setText(file);
}

void MainWindow::on_modelfile_browse_clicked()
{
    QString file, dir;
    dir = dirname(ui->modelfile->text().toStdString());
    file = QFileDialog::getOpenFileName(this, QString("Select model file..."), dir, QString("*.xml"));
    if ( !file.isEmpty() )
        ui->modelfile->setText(file);
}

void MainWindow::on_simparams_reset_clicked()
{
    ui->outdir->setText(QString::fromStdString(config.output.dir));
    ui->dt->setValue(config.io.dt);
    ui->modelfile->setText(QString::fromStdString(config.model.deffile));
}

void MainWindow::on_simparams_apply_clicked()
{
    std::string mfile = ui->modelfile->text().toStdString();
    double dt = ui->dt->value();

    config.model.deffile = mfile;
    config.io.dt = dt;
    config.output.dir = ui->outdir->text().toStdString();
}

void MainWindow::on_vclamp_start_clicked()
{
    run_vclamp_start();
}

void MainWindow::on_vclamp_stop_clicked()
{
    run_vclamp_stop();
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
