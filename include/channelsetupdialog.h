/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-04

--------------------------------------------------------------------------*/
#ifndef CHANNELSETUPDIALOG_H
#define CHANNELSETUPDIALOG_H

#include <QDialog>
#include <QAbstractButton>
#include <QDataWidgetMapper>
#include <QStringListModel>
#include <QStandardItemModel>
#include <QItemDelegate>
#include "channeleditormodel.h"
#include "devicechannelmodel.h"
#include "devicerangemodel.h"
#include "devicereferencemodel.h"
#include "channellistmodel.h"

namespace Ui {
class ChannelSetupDialog;
}

class ChannelSetupDialog : public QDialog
{
    Q_OBJECT

public:
    explicit ChannelSetupDialog(QWidget *parent = 0);
    ~ChannelSetupDialog();

public slots:
    void open();
    void addChannel();
    void removeChannel();
    void selectionChanged(QModelIndex index, QModelIndex previous);

signals:
    void channelsUpdated();

private:
    Ui::ChannelSetupDialog *ui;
    ChannelEditorModel *editor;
    QDataWidgetMapper *mapper;
    DeviceChannelModel *channelModel;
    DeviceRangeModel *rangeModel;
    DeviceReferenceModel *arefModel;
    ChannelListModel *chanList;
    ChannelListModel *offsetSources;
};

class ComboboxDataDelegate : public QItemDelegate
{
    Q_OBJECT
public:
    ComboboxDataDelegate(QObject *parent = 0);
    void setEditorData(QWidget *editor, const QModelIndex &index) const;
    void setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const;
};

#endif // CHANNELSETUPDIALOG_H
