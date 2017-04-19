#ifndef PROFILEDIALOG_H
#define PROFILEDIALOG_H

#include <QDialog>
#include "errorprofiler.h"
#include "wavegendialog.h"
#include "session.h"

namespace Ui {
class ProfileDialog;
}
class QSpinBox;
class QDoubleSpinBox;

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
    void aborted();

    void updateCombo();
    void updatePresets();

    void on_btnStart_clicked();

    void on_btnAbort_clicked();

    void on_btnPreset_clicked();

private:
    Ui::ProfileDialog *ui;
    Session &session;

    std::vector<QSpinBox*> ns;
    std::vector<QDoubleSpinBox*> mins, maxes;

    static constexpr int nHardPresets = 6;

    void setCloseRange(int i);
};

#endif // PROFILEDIALOG_H
