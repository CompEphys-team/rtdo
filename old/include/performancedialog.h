/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-02-04

--------------------------------------------------------------------------*/
#ifndef PERFORMANCEDIALOG_H
#define PERFORMANCEDIALOG_H

#include <QDialog>
#include <QAbstractButton>

namespace Ui {
class PerformanceDialog;
}

class PerformanceDialog : public QDialog
{
    Q_OBJECT

public:
    explicit PerformanceDialog(QWidget *parent = 0);
    ~PerformanceDialog();

public slots:
    void open();
    void accept();

private slots:
    void on_buttonBox_clicked(QAbstractButton *button);
    void apply();

    void on_reportingToggle_clicked();

private:
    Ui::PerformanceDialog *ui;
};

#endif // PERFORMANCEDIALOG_H
