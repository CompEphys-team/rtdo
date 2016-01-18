#include "include/modelsetupdialog.h"
#include "ui_modelsetupdialog.h"
#include <QFileDialog>
#include "config.h"
#include "util.h"

ModelSetupDialog::ModelSetupDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ModelSetupDialog)
{
    ui->setupUi(this);
}

ModelSetupDialog::~ModelSetupDialog()
{
    delete ui;
}

void ModelSetupDialog::open()
{
    ui->modelfile->setText(QString::fromStdString(config->model.deffile));
    ui->outdir->setText(QString::fromStdString(config->output.dir));
    ui->dt->setValue(config->io.dt);
    QDialog::open();
}

void ModelSetupDialog::accept()
{
    config->model.deffile = ui->modelfile->text().toStdString();
    config->output.dir = ui->outdir->text().toStdString();
    config->io.dt = ui->dt->value();
    QDialog::accept();
}

void ModelSetupDialog::on_modelfile_browse_clicked()
{
    QString file = QFileDialog::getOpenFileName(this,
                                                QString("Select model file..."),
                                                dirname(ui->outdir->text()),
                                                QString("*.xml"));
    if ( !file.isEmpty() )
        ui->modelfile->setText(file);
}

void ModelSetupDialog::on_outdir_browse_clicked()
{
    QString file = QFileDialog::getExistingDirectory(this,
                                                     QString("Select data output directory..."),
                                                     dirname(ui->outdir->text()));
    if ( !file.isEmpty() )
        ui->outdir->setText(file);
}
