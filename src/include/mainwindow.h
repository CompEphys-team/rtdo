#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QThread>
#include "wavegendialog.h"
#include "profiledialog.h"
#include "metamodel.h"

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

private:
    Ui::MainWindow *ui;
    WavegenDialog *wavegenDlg;
    ProfileDialog *profileDlg;

    MetaModel model;

    QThread gthread;

    void closeEvent(QCloseEvent *event);

    ExperimentLibrary &getExpLib();
    ExperimentLibrary *expLib;
};

#endif // MAINWINDOW_H
