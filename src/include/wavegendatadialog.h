#ifndef WAVEGENDATADIALOG_H
#define WAVEGENDATADIALOG_H

#include <QDialog>
#include "session.h"

namespace Ui {
class WavegenDataDialog;
}
class QAbstractButton;

class WavegenDataDialog : public QDialog
{
    Q_OBJECT

public:
    explicit WavegenDataDialog(Session &s, int historicIndex = -1, QWidget *parent = 0);
    ~WavegenDataDialog();

public slots:
    void importData();
    void exportData();

signals:
    void apply(WavegenData);

private slots:
    void on_buttonBox_clicked(QAbstractButton *button);

private:
    Ui::WavegenDataDialog *ui;
    Session &session;
    int historicIndex;
};

#endif // WAVEGENDATADIALOG_H
