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
    explicit ProfileDialog(Session *s, QWidget *parent = 0);
    ~ProfileDialog();

    void selectionsChanged(WavegenDialog *dlg);

signals:
    void profile();

private slots:
    void profileComplete(int);
    void done();

    void on_btnStart_clicked();

    void on_btnAbort_clicked();

    void on_btnReplot_clicked();

private:
    Ui::ProfileDialog *ui;
    Session *session;

    ErrorProfiler &profiler;

    std::list<QString> actions;

    std::vector<WavegenDialog::Selection> *selections;
    WavegenDialog::Selection selection;
};

#endif // PROFILEDIALOG_H
