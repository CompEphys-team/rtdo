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
#include "softrtdaq.h"
#include "globals.h"
#include "rt.h"
#include "run.h"
#include <QFileDialog>
#include <comedilib.h>

#define AREF_TEXT_GROUND "Ground"
#define AREF_TEXT_COMMON "Common"
#define AREF_TEXT_DIFF   "Diff"
#define AREF_TEXT_OTHER  "Other"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    this->load_channel_setup();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::setup_channel_tab(QComboBox *ch, QComboBox *range, QComboBox *ref, const daq_channel *chan) {
    int i, k, flags;
    QString str;

    ch->clear();
    for ( i = 0, k = comedi_get_n_channels(chan->device, chan->subdevice); i < k; i++ ) {
        str.sprintf("%d", i);
        ch->addItem(str, Qt::DisplayRole);
    }

    reload_ranges(range, chan, 0);

    flags = comedi_get_subdevice_flags(chan->device, chan->subdevice);
    if ( flags & SDF_GROUND ) {
        ref->addItem(AREF_TEXT_GROUND, Qt::DisplayRole);
    }
    if ( flags & SDF_COMMON ) {
        ref->addItem(AREF_TEXT_COMMON, Qt::DisplayRole);
    }
    if ( flags & SDF_DIFF ) {
        ref->addItem(AREF_TEXT_DIFF, Qt::DisplayRole);
    }
    if ( flags & SDF_OTHER ) {
        ref->addItem(AREF_TEXT_OTHER, Qt::DisplayRole);
    }
}

void MainWindow::load_channel_setup()
{
    setup_channel_tab(ui->vout_chan, ui->vout_range, ui->vout_reference, &daqchan_vout);
    setup_channel_tab(ui->cout_chan, ui->cout_range, ui->cout_reference, &daqchan_cout);
    setup_channel_tab(ui->vin_chan, ui->vin_range, ui->vin_reference, &daqchan_vin);
    setup_channel_tab(ui->cin_chan, ui->cin_range, ui->cin_reference, &daqchan_cin);
    on_channels_reset_clicked();
    on_simparams_reset_clicked();
}

void MainWindow::on_vout_chan_currentIndexChanged(int index)
{
    static int flags = comedi_get_subdevice_flags(daqchan_vout.device, daqchan_vout.subdevice);
    if ( flags & SDF_RANGETYPE )
        reload_ranges(ui->vout_range, &daqchan_vout, index);
}

void MainWindow::on_cout_chan_currentIndexChanged(int index)
{
    static int flags = comedi_get_subdevice_flags(daqchan_cout.device, daqchan_cout.subdevice);
    if ( flags & SDF_RANGETYPE )
        reload_ranges(ui->cout_range, &daqchan_cout, index);
}

void MainWindow::on_vin_chan_currentIndexChanged(int index)
{
    static int flags = comedi_get_subdevice_flags(daqchan_vin.device, daqchan_vin.subdevice);
    if ( flags & SDF_RANGETYPE )
        reload_ranges(ui->vin_range, &daqchan_vin, index);
}

void MainWindow::on_cin_chan_currentIndexChanged(int index)
{
    static int flags = comedi_get_subdevice_flags(daqchan_cin.device, daqchan_cin.subdevice);
    if ( flags & SDF_RANGETYPE )
        reload_ranges(ui->cin_range, &daqchan_cin, index);
}

void MainWindow::reload_ranges(QComboBox *el, const daq_channel *chan, unsigned int channel)
{
    int i, k, idx = el->currentIndex();
    QString str;
    comedi_range *r;
    el->clear();
    for ( i = 0, k = comedi_get_n_ranges(chan->device, chan->subdevice, channel); i < k; i++ ) {
        r = comedi_get_range(chan->device, chan->subdevice, channel, i);
        str.sprintf("[%.1f, %.1f]", r->min, r->max);
        el->addItem(str, Qt::DisplayRole);
    }
    if ( idx < 0 || idx > el->count())
        idx = 0;
    el->setCurrentIndex(idx);
}

void MainWindow::on_channels_apply_clicked()
{
    channels_apply_generic( &daqchan_vout,
                ui->vout_chan->currentIndex(),
                ui->vout_range->currentIndex(),
                ui->vout_reference->currentText(),
                ui->vout_gain->value(),
                ui->vout_offset->value());
    channels_apply_generic( &daqchan_cout,
                ui->cout_chan->currentIndex(),
                ui->cout_range->currentIndex(),
                ui->cout_reference->currentText(),
                ui->cout_gain->value(),
                ui->cout_offset->value());
    channels_apply_generic( &daqchan_vin,
                ui->vin_chan->currentIndex(),
                ui->vin_range->currentIndex(),
                ui->vin_reference->currentText(),
                ui->vin_gain->value(),
                ui->vin_offset->value());
    channels_apply_generic( &daqchan_cin,
                ui->cin_chan->currentIndex(),
                ui->cin_range->currentIndex(),
                ui->cin_reference->currentText(),
                ui->cin_gain->value(),
                ui->cin_offset->value());
}

void MainWindow::channels_apply_generic(
        daq_channel *chan, int ichan, int irange, QString aref, double gain, double offset)
{
    chan->channel = (unsigned short)ichan;
    chan->range = (unsigned short)irange;
    chan->gain = gain;
    chan->offset = offset;
    if ( aref == AREF_TEXT_GROUND ) chan->aref = AREF_GROUND;
    if ( aref == AREF_TEXT_COMMON ) chan->aref = AREF_COMMON;
    if ( aref == AREF_TEXT_DIFF )   chan->aref = AREF_DIFF;
    if ( aref == AREF_TEXT_OTHER )  chan->aref = AREF_OTHER;
    daq_create_converter(chan);
}

void MainWindow::on_channels_reset_clicked()
{
    channels_reset_generic(&daqchan_vout,
                ui->vout_chan,
                ui->vout_range,
                ui->vout_reference,
                ui->vout_gain,
                ui->vout_offset);
    channels_reset_generic(&daqchan_cout,
                ui->cout_chan,
                ui->cout_range,
                ui->cout_reference,
                ui->cout_gain,
                ui->cout_offset);
    channels_reset_generic(&daqchan_vin,
                ui->vin_chan,
                ui->vin_range,
                ui->vin_reference,
                ui->vin_gain,
                ui->vin_offset);
    channels_reset_generic(&daqchan_cin,
                ui->cin_chan,
                ui->cin_range,
                ui->cin_reference,
                ui->cin_gain,
                ui->cin_offset);
}

void MainWindow::channels_reset_generic(
        daq_channel *chan, QComboBox *ichan, QComboBox *range,
        QComboBox *aref, QDoubleSpinBox *gain, QDoubleSpinBox *offset)
{
    ichan->setCurrentIndex(chan->channel);
    range->setCurrentIndex(chan->range);
    gain->setValue(chan->gain);
    offset->setValue(chan->offset);
    QString ref;
    if ( chan->aref == AREF_GROUND ) ref = AREF_TEXT_GROUND;
    if ( chan->aref == AREF_COMMON ) ref = AREF_TEXT_COMMON;
    if ( chan->aref == AREF_DIFF )   ref = AREF_TEXT_DIFF;
    if ( chan->aref == AREF_OTHER )  ref = AREF_TEXT_OTHER;
    int index = aref->findText(ref);
    aref->setCurrentIndex(index);
}

void MainWindow::on_vout_offset_read_clicked()
{
    int err=0;
    double val = rtdo_read_now(daqchan_vin.handle, &err);
    if ( !err )
        ui->vout_offset->setValue(val);
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
    ui->outdir->setText(QString::fromStdString(sim_params.outdir));
    ui->sigfile->setText(QString::fromStdString(sim_params.sigfile));
    ui->vc_waveforms->setText(QString::fromStdString(sim_params.vc_wavefile));
    ui->dt->setValue(sim_params.dt);
    ui->npop->setValue(sim_params.nPop);
    ui->modelfile->setText(QString::fromStdString(sim_params.modelfile));
}

void MainWindow::on_simparams_apply_clicked()
{
    std::string mfile = ui->modelfile->text().toStdString();
    double dt = ui->dt->value();
    int npop = ui->npop->value();
    bool recompile = sim_params.modelfile.compare(mfile)
            || sim_params.dt != dt
            || sim_params.nPop != npop;

    // Compile-time parameters
    sim_params.modelfile = mfile;
    sim_params.dt = dt;
    sim_params.nPop = npop;

    // Runtime parameters
    sim_params.outdir = ui->outdir->text().toStdString();
    sim_params.sigfile = ui->sigfile->text().toStdString();
    sim_params.vc_wavefile = ui->vc_waveforms->text().toStdString();

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
