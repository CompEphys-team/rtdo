/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-04

--------------------------------------------------------------------------*/
#ifndef DEVICECHANNELMODEL_H
#define DEVICECHANNELMODEL_H

#include <QAbstractListModel>
#include <QDataWidgetMapper>
#include "channeleditormodel.h"

class DeviceChannelModel : public QAbstractListModel
{
    Q_OBJECT
public:
    explicit DeviceChannelModel(ChannelEditorModel *editor, QDataWidgetMapper *mapper, QObject *parent = 0);
    QVariant data(const QModelIndex &index, int role) const;
    int rowCount(const QModelIndex &parent = QModelIndex()) const;

protected:
    ChannelEditorModel *editor;
    QDataWidgetMapper *mapper;
};

#endif // DEVICECHANNELMODEL_H
