/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2015-12-03

--------------------------------------------------------------------------*/
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QComboBox>
#include <QDoubleSpinBox>
#include "types.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    void lock_channels(bool lock);

private slots:
    void on_vout_chan_currentIndexChanged(int index);
    void on_cout_chan_currentIndexChanged(int index);
    void on_vin_chan_currentIndexChanged(int index);
    void on_cin_chan_currentIndexChanged(int index);

    void on_channels_apply_clicked();

    void on_channels_reset_clicked();

    void on_vout_offset_read_clicked();

    void on_vc_waveforms_browse_clicked();

    void on_sigfile_browse_clicked();

    void on_outdir_browse_clicked();

    void on_simparams_reset_clicked();

    void on_simparams_apply_clicked();

    void on_vclamp_start_clicked();

    void on_vclamp_stop_clicked();

private:
    Ui::MainWindow *ui;

    void load_channel_setup();
    void reload_ranges(QComboBox *el, int channel, enum comedi_subdevice_type subdevice_type);
    void channels_apply_generic(daq_channel *chan, int ichan, int irange, QString aref, double gain, double offset);
    void channels_reset_generic(daq_channel *chan, QComboBox *ichan, QComboBox *range,
                                QComboBox *aref, QDoubleSpinBox *gain, QDoubleSpinBox *offset);

    bool ai_rangetype;
    bool ao_rangetype;
    bool channels_locked;
};

#endif // MAINWINDOW_H
