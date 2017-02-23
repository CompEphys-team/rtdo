#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "wavegendialog.h"
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

private:
    Ui::MainWindow *ui;
    WavegenDialog *wavegenDlg;

    MetaModel model;
};

#endif // MAINWINDOW_H
