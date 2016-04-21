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
#include <memory>
#include "channelsetupdialog.h"
#include "vclampsetupdialog.h"
#include "wavegensetupdialog.h"
#include "modelsetupdialog.h"
#include "performancedialog.h"
#include "runner.h"
#include "module.h"
#include "actionlistmodel.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

public slots:
    void updateConfigFields();

signals:
    void configChanged();

private slots:
    // pSetup
    void compile();
    void on_actionSave_configuration_triggered();
    void on_actionLoad_configuration_triggered();

    // pWavegen
    void on_wavegen_start_clicked();
    void on_wavegen_start_NS_clicked();
    void wavegenComplete(bool successfully);

    // pExperiment
    void qAction(QAction *action);
    void actionComplete(int handle);
    void on_btnQRemove_clicked();
    void on_btnQStart_clicked();
    void on_btnQSkip_clicked();

    void on_VCApply_clicked();
    void zeroOutputs();

    bool pExpInit();
    void on_pExperimentReset_clicked();
    void outdirSet();

    void on_btnNotesLoad_clicked();
    void on_btnNotesSave_clicked();

    void offlineAction(QAction *action);

    // page transitions
    void on_pSetup2Experiment_clicked();
    void on_pSetup2Wavegen_clicked();
    void on_pWavegen2Setup_clicked();
    void on_pExperiment2Setup_clicked();

    void pExp2Setup();

private:
    Ui::MainWindow *ui;
    unique_ptr<ChannelSetupDialog> channel_setup;
    unique_ptr<VClampSetupDialog> vclamp_setup;
    unique_ptr<WavegenSetupDialog> wavegen_setup;
    unique_ptr<ModelSetupDialog> model_setup;
    unique_ptr<PerformanceDialog> performance;

    unique_ptr<CompileRunner> compiler;
    unique_ptr<Runner> wavegen;
    unique_ptr<Runner> wavegenNS;

    Module *module;
    ActionListModel *protocol;

    bool offlineNoAsk;
};

#endif // MAINWINDOW_H
