#ifndef WAVEGENDIALOG_H
#define WAVEGENDIALOG_H

#include <QDialog>
#include "session.h"

namespace Ui {
class WavegenDialog;
}

class WavegenDialog : public QDialog
{
    Q_OBJECT

public:
    explicit WavegenDialog(Session &s, QWidget *parent = 0);
    ~WavegenDialog();

signals:
    void adjustSigmas();
    void search(int param);

private slots:
    void end(int);
    void startedSearch(int);
    void searchTick(int);

private:
    Ui::WavegenDialog *ui;
    Session &session;

    std::list<QString> actions;

    bool abort;
};

#endif // WAVEGENDIALOG_H
