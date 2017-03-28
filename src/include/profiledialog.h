#ifndef PROFILEDIALOG_H
#define PROFILEDIALOG_H

#include <QDialog>
#include "errorprofiler.h"
#include "wavegendialog.h"
#include "session.h"

namespace Ui {
class ProfileDialog;
}

class ProfileDialog : public QDialog
{
    Q_OBJECT

public:
    explicit ProfileDialog(Session &s, QWidget *parent = 0);
    ~ProfileDialog();

signals:
    void profile();

private slots:
    void profileComplete(int);
    void done();

    void updateCombo();
    void updateRange();

    void on_btnStart_clicked();

    void on_btnAbort_clicked();

private:
    Ui::ProfileDialog *ui;
    Session &session;

    int numStimulations;
};

#endif // PROFILEDIALOG_H
