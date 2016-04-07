/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-20

--------------------------------------------------------------------------*/
#ifndef VCLAMPSETUPDIALOG_H
#define VCLAMPSETUPDIALOG_H

#include <QDialog>
#include "channellistmodel.h"

namespace Ui {
class VClampSetupDialog;
}

class VClampSetupDialog : public QDialog
{
    Q_OBJECT

public:
    explicit VClampSetupDialog(QWidget *parent = 0);
    ~VClampSetupDialog();

signals:
    void channelsUpdated();
    void configChanged();

public slots:
    void open();
    void accept();

private slots:
    void on_waveformBrowse_clicked();

private:
    Ui::VClampSetupDialog *ui;
    ChannelListModel *cinModel;
    ChannelListModel *voutModel;
};

#endif // VCLAMPSETUPDIALOG_H
