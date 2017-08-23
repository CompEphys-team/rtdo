#ifndef RUNDATADIALOG_H
#define RUNDATADIALOG_H

#include <QDialog>
#include "session.h"
#include "calibrator.h"

namespace Ui {
class RunDataDialog;
}
class QAbstractButton;

class RunDataDialog : public QDialog
{
    Q_OBJECT

public:
    explicit RunDataDialog(Session &s, QWidget *parent = 0);
    ~RunDataDialog();

public slots:
    void importData();
    void exportData();

private slots:
    void on_buttonBox_accepted();
    void on_buttonBox_rejected();

private:
    Ui::RunDataDialog *ui;
    Session &session;
    Calibrator calibrator;

signals:
    void apply(RunData);
};

#endif // RUNDATADIALOG_H
