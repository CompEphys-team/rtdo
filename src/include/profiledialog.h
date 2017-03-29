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
    void generate();

private slots:
    void profileProgress(int, int);
    void done();

    void updateCombo();
    void updateRange();

    void on_btnStart_clicked();

    void on_btnAbort_clicked();

private:
    Ui::ProfileDialog *ui;
    Session &session;
};

#endif // PROFILEDIALOG_H
