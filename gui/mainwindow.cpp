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

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    channel_setup(new ChannelSetupDialog)
{
    ui->setupUi(this);
    on_simparams_reset_clicked();
}

MainWindow::~MainWindow()
{
    delete ui;
}

QString dirname(std::string path) {
    int lastslash = path.find_last_of('/');
    if ( lastslash )
        return QString::fromStdString(path.substr(0, lastslash));
    else
        return QString();
}

void MainWindow::on_vc_waveforms_browse_clicked()
{
    QString file, dir;
    dir = dirname(ui->vc_waveforms->text().toStdString());
    file = QFileDialog::getOpenFileName(this, QString("Select voltage clamp waveform file..."), dir);
    if ( !file.isEmpty() )
        ui->vc_waveforms->setText(file);
}

void MainWindow::on_sigfile_browse_clicked()
{
    QString file, dir;
    dir = dirname(ui->sigfile->text().toStdString());
    file = QFileDialog::getOpenFileName(this, QString("Select parameter file..."), dir);
    if ( !file.isEmpty() )
        ui->sigfile->setText(file);
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
    file = QFileDialog::getOpenFileName(this, QString("Select model file..."), dir, QString("C++ Files (*.cc *.cpp)"));
    if ( !file.isEmpty() )
        ui->modelfile->setText(file);
}

void MainWindow::on_simparams_reset_clicked()
{
    ui->outdir->setText(QString::fromStdString(config.output.dir));
    ui->sigfile->setText(QString::fromStdString(config.vc.sigfile));
    ui->vc_waveforms->setText(QString::fromStdString(config.vc.wavefile));
    ui->dt->setValue(config.io.dt);
    ui->npop->setValue(config.vc.popsize);
    ui->modelfile->setText(QString::fromStdString(config.model.deffile));
}

void MainWindow::on_simparams_apply_clicked()
{
    std::string mfile = ui->modelfile->text().toStdString();
    double dt = ui->dt->value();
    int npop = ui->npop->value();
    bool recompile = config.model.deffile.compare(mfile)
            || config.io.dt != dt
            || config.vc.popsize != npop;

    // Compile-time parameters
    config.model.deffile = mfile;
    config.io.dt = dt;
    config.vc.popsize = npop;

    // Runtime parameters
    config.output.dir = ui->outdir->text().toStdString();
    config.vc.sigfile = ui->sigfile->text().toStdString();
    config.vc.wavefile = ui->vc_waveforms->text().toStdString();

    if ( recompile ) {
        QApplication::setOverrideCursor(QCursor(Qt::BusyCursor));
        compile_model();
        QApplication::restoreOverrideCursor();
    }
}

void MainWindow::on_vclamp_start_clicked()
{
    run_vclamp_start();
}

void MainWindow::on_vclamp_stop_clicked()
{
    run_vclamp_stop();
}

void MainWindow::on_actionChannel_setup_triggered()
{
    channel_setup->open();
}
