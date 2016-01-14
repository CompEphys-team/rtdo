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
