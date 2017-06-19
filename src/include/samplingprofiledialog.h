#ifndef SAMPLINGPROFILEDIALOG_H
#define SAMPLINGPROFILEDIALOG_H

#include <QWidget>
#include "samplingprofiler.h"
#include "session.h"

namespace Ui {
class SamplingProfileDialog;
}
class QDoubleSpinBox;

class SamplingProfileDialog : public QWidget
{
    Q_OBJECT

public:
    explicit SamplingProfileDialog(Session &s, QWidget *parent = 0);
    ~SamplingProfileDialog();

signals:
    void generate(SamplingProfiler::Profile);

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
    Ui::SamplingProfileDialog *ui;
    Session &session;

    std::vector<QDoubleSpinBox*> mins, maxes;

    static constexpr int nHardPresets = 2;

    void setCloseRange(int i);

private:
};

#endif // SAMPLINGPROFILEDIALOG_H
