#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include "wavegendialog.h"
#include "profiledialog.h"
#include "samplingprofiledialog.h"
#include "deckwidget.h"
#include "gafitterwidget.h"
#include "project.h"
#include "session.h"
#include "projectsettingsdialog.h"
#include "scope.h"

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
    void on_actionWavegen_triggered();
    void on_actionProfiler_triggered();
    void on_actionDecks_triggered();
    void on_actionGAFitter_triggered();

    void on_actionNew_project_triggered();

    void on_actionOpen_project_triggered();

    void on_actionNew_session_triggered();

    void on_actionOpen_session_triggered();

    void on_actionWavegen_fitness_map_triggered();

    void on_actionError_profiles_triggered();

    void on_actionFitting_Parameters_triggered();

    void on_actionStimulations_triggered();

    void setTitle();

    void on_actionStimulation_editor_triggered();

    void on_actionGA_Fitter_triggered();

    void on_actionSampling_profiler_triggered();

    void on_actionSampled_profiles_triggered();

    void on_actionRunData_triggered();

    void on_actionWavegenData_triggered();

    void on_actionStimulationData_triggered();

    void on_actionDAQData_triggered();

    void on_actionCrossload_from_other_session_triggered();

    void on_runCtrl_activated(int index);

    void on_abort_clicked();

    void on_remove_clicked();

    void on_actionScope_triggered();

private:
    Ui::MainWindow *ui;
    WavegenDialog *wavegenDlg;
    ProfileDialog *profileDlg;
    std::unique_ptr<DeckWidget> deckWidget;
    std::unique_ptr<GAFitterWidget> gaFitterWidget;
    std::unique_ptr<SamplingProfileDialog> sprofileDlg;
    std::unique_ptr<ProjectSettingsDialog> projectSettingsDlg;
    std::unique_ptr<Scope> scope;

    Project *project;
    Session *session;

    QString title;
    QLabel *workerStatus;

    void closeEvent(QCloseEvent *event);
    void sessionOpened();
};

#endif // MAINWINDOW_H
