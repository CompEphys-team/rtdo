#ifndef WAVEGENDATADIALOG_H
#define WAVEGENDATADIALOG_H

#include <QDialog>
#include "session.h"

namespace Ui {
class WavegenDataDialog;
}

class WavegenDataDialog : public QDialog
{
    Q_OBJECT

public:
    explicit WavegenDataDialog(Session &s, QWidget *parent = 0);
    ~WavegenDataDialog();

public slots:
    void importData();
    void exportData();

signals:
    void apply(WavegenData);

private:
    Ui::WavegenDataDialog *ui;
    Session &session;
};

#endif // WAVEGENDATADIALOG_H
