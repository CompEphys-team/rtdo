#ifndef CANNEDCHANNELASSOCIATIONDIALOG_H
#define CANNEDCHANNELASSOCIATIONDIALOG_H

#include <QDialog>
#include "session.h"
#include "canneddaq.h"

namespace Ui {
class CannedChannelAssociationDialog;
}

class CannedChannelAssociationDialog : public QDialog
{
    Q_OBJECT

public:
    explicit CannedChannelAssociationDialog(Session &s, CannedDAQ *daq, QWidget *parent = 0);
    ~CannedChannelAssociationDialog();

private slots:
    void on_CannedChannelAssociationDialog_accepted();

private:
    Ui::CannedChannelAssociationDialog *ui;
    Session &session;
    CannedDAQ *daq;
};

#endif // CANNEDCHANNELASSOCIATIONDIALOG_H
