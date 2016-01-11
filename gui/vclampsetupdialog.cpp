#include "vclampsetupdialog.h"
#include "ui_vclampsetupdialog.h"
#include "util.h"
#include <QFileDialog>
#include "config.h"
#include "run.h"

VClampSetupDialog::VClampSetupDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::VClampSetupDialog),
    cinModel(new ChannelListModel(ChannelListModel::AnalogIn)),
    voutModel(new ChannelListModel(ChannelListModel::AnalogOut))
{
    ui->setupUi(this);
    ui->currentInputChannel->setModel(cinModel);
    ui->voltageOutputChannel->setModel(voutModel);
    connect(this, SIGNAL(channelsUpdated()), cinModel, SIGNAL(modelReset()));
    connect(this, SIGNAL(channelsUpdated()), voutModel, SIGNAL(modelReset()));
}

VClampSetupDialog::~VClampSetupDialog()
{
    delete ui;
    delete cinModel;
    delete voutModel;
}

void VClampSetupDialog::open()
{
    vector<daq_channel *>::iterator it;
    int i;
    ui->currentInputChannel->setCurrentIndex(-1);
    ui->voltageOutputChannel->setCurrentIndex(-1);
    for ( it = config.io.channels.begin(), i = 0; it != config.io.channels.end(); ++it, ++i ) {
        if ( *it == config.vc.in )
            ui->currentInputChannel->setCurrentIndex(i);
        if ( *it == config.vc.out )
            ui->voltageOutputChannel->setCurrentIndex(i);
    }

    ui->waveformFile->setText(QString::fromStdString(config.vc.wavefile));
    ui->sigmaFile->setText(QString::fromStdString(config.vc.sigfile));
    ui->popSize->setValue(config.vc.popsize);

    QDialog::open();
}

void VClampSetupDialog::accept()
{
    config.vc.in = ( ui->currentInputChannel->currentIndex() >= 0 )
            ? *(config.io.channels.begin() + ui->currentInputChannel->currentIndex())
            : 0;
    config.vc.out = ( ui->voltageOutputChannel->currentIndex() >= 0 )
            ? *(config.io.channels.begin() + ui->voltageOutputChannel->currentIndex())
            : 0;
    config.vc.wavefile = ui->waveformFile->text().toStdString();
    config.vc.sigfile = ui->sigmaFile->text().toStdString();
    config.vc.popsize = ui->popSize->value();

    QDialog::accept();

    QApplication::setOverrideCursor(QCursor(Qt::BusyCursor));
    compile_model();
    QApplication::restoreOverrideCursor();
}

void VClampSetupDialog::on_waveformBrowse_clicked()
{
    QString file, dir;
    dir = dirname(ui->waveformFile->text());
    file = QFileDialog::getOpenFileName(this, QString("Select voltage clamp waveform file..."), dir);
    if ( !file.isEmpty() )
        ui->waveformFile->setText(file);
}

void VClampSetupDialog::on_sigmaBrowse_clicked()
{
    QString file, dir;
    dir = dirname(ui->sigmaFile->text());
    file = QFileDialog::getOpenFileName(this, QString("Select voltage clamp sigma file..."), dir);
    if ( !file.isEmpty() )
        ui->sigmaFile->setText(file);
}
