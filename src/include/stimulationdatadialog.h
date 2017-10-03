#ifndef STIMULATIONDATADIALOG_H
#define STIMULATIONDATADIALOG_H

#include <QDialog>
#include "session.h"

namespace Ui {
class StimulationDataDialog;
}

class StimulationDataDialog : public QDialog
{
    Q_OBJECT

public:
    explicit StimulationDataDialog(Session &s, QWidget *parent = 0);
    ~StimulationDataDialog();

public slots:
    void importData();
    void exportData();

signals:
    void apply(StimulationData);
    void updateWavegenData(WavegenData);

private:
    Ui::StimulationDataDialog *ui;
    Session &session;
};

#endif // STIMULATIONDATADIALOG_H
