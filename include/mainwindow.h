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
#include "channelsetupdialog.h"
#include "vclampsetupdialog.h"
#include "wavegensetupdialog.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_outdir_browse_clicked();

    void on_modelfile_browse_clicked();

    void on_simparams_reset_clicked();

    void on_simparams_apply_clicked();

    void on_vclamp_start_clicked();

    void on_vclamp_stop_clicked();

    void on_actionSave_configuration_triggered();

    void on_actionLoad_configuration_triggered();

    void on_wavegen_start_clicked();

    void on_wavegen_stop_clicked();

    void on_wavegen_compile_clicked();

private:
    Ui::MainWindow *ui;
    ChannelSetupDialog *channel_setup;
    VClampSetupDialog *vclamp_setup;
    WavegenSetupDialog *wavegen_setup;
};

#endif // MAINWINDOW_H
