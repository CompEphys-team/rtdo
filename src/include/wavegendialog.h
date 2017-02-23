#ifndef WAVEGENDIALOG_H
#define WAVEGENDIALOG_H

#include <QDialog>
#include "wavegen.h"

namespace Ui {
class WavegenDialog;
}

class WavegenDialog : public QDialog
{
    Q_OBJECT

public:
    explicit WavegenDialog(MetaModel &model, QWidget *parent = 0);
    ~WavegenDialog();

private:
    Ui::WavegenDialog *ui;

    MetaModel &model;
    WavegenLibrary lib;
    Wavegen *wg;
};

#endif // WAVEGENDIALOG_H
