/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-20

--------------------------------------------------------------------------*/
#ifndef WAVEGENSETUPDIALOG_H
#define WAVEGENSETUPDIALOG_H

#include <QDialog>

namespace Ui {
class WavegenSetupDialog;
}

class WavegenSetupDialog : public QDialog
{
    Q_OBJECT

public:
    explicit WavegenSetupDialog(QWidget *parent = 0);
    ~WavegenSetupDialog();

public slots:
    void open();
    void accept();

private:
    Ui::WavegenSetupDialog *ui;
};

#endif // WAVEGENSETUPDIALOG_H
