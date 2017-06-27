#ifndef RUNDATADIALOG_H
#define RUNDATADIALOG_H

#include <QWidget>
#include "session.h"

namespace Ui {
class RunDataDialog;
}
class QAbstractButton;

class RunDataDialog : public QWidget
{
    Q_OBJECT

public:
    explicit RunDataDialog(Session &s, QWidget *parent = 0);
    ~RunDataDialog();

    void importData();
    void exportData();

private slots:
    void on_buttonBox_accepted();
    void on_buttonBox_rejected();

private:
    Ui::RunDataDialog *ui;
    Session &session;

signals:
    void apply(RunData);
};

#endif // RUNDATADIALOG_H
