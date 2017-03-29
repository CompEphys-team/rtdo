#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "wavegendialog.h"
#include "profiledialog.h"
#include "project.h"
#include "session.h"

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

    void on_actionNew_project_triggered();

    void on_actionOpen_project_triggered();

    void on_actionNew_session_triggered();

    void on_actionOpen_session_triggered();

    void on_actionWavegen_fitness_map_triggered();

    void on_actionError_profiles_triggered();

private:
    Ui::MainWindow *ui;
    WavegenDialog *wavegenDlg;
    ProfileDialog *profileDlg;

    Project *project;
    Session *session;

    void closeEvent(QCloseEvent *event);
};

#endif // MAINWINDOW_H
