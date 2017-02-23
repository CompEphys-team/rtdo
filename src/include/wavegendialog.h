#ifndef WAVEGENDIALOG_H
#define WAVEGENDIALOG_H

#include <QDialog>
#include <QThread>
#include "wavegen.h"

namespace Ui {
class WavegenDialog;
}

class WavegenDialog : public QDialog
{
    Q_OBJECT

public:
    explicit WavegenDialog(MetaModel &model, QThread *thread, QWidget *parent = 0);
    ~WavegenDialog();

signals:
    void permute();
    void adjustSigmas();
    void search(int param);

private slots:
    void end(int);
    void startedSearch(int);
    void searchTick(int);

private:
    Ui::WavegenDialog *ui;
    QThread *thread;

    MetaModel &model;
    WavegenLibrary lib;
    Wavegen *wg;

    std::list<QString> actions;
};

#endif // WAVEGENDIALOG_H
