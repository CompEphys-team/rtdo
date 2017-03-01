#ifndef PROFILEDIALOG_H
#define PROFILEDIALOG_H

#include <QDialog>
#include "errorprofiler.h"
#include "wavegendialog.h"

namespace Ui {
class ProfileDialog;
}

class ProfileDialog : public QDialog
{
    Q_OBJECT

public:
    explicit ProfileDialog(ExperimentLibrary &lib, QThread *thread, QWidget *parent = 0);
    ~ProfileDialog();

    void selectionsChanged(WavegenDialog *dlg);

private:
    Ui::ProfileDialog *ui;
    QThread *thread;

    ExperimentLibrary &lib;
    ErrorProfiler profiler;

    std::vector<WavegenDialog::Selection> *selections;
};

#endif // PROFILEDIALOG_H
