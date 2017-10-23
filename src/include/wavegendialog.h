#ifndef WAVEGENDIALOG_H
#define WAVEGENDIALOG_H

#include <QDialog>
#include "session.h"

namespace Ui {
class WavegenDialog;
}

class WavegenDialog : public QDialog
{
    Q_OBJECT

public:
    explicit WavegenDialog(Session &s, QWidget *parent = 0);
    ~WavegenDialog();

private:
    Ui::WavegenDialog *ui;
    Session &session;
};

#endif // WAVEGENDIALOG_H
