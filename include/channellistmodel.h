/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-04

--------------------------------------------------------------------------*/
#ifndef CHANNELLISTMODEL_H
#define CHANNELLISTMODEL_H

#include <QAbstractListModel>
#include "types.h"

class ChannelListModel : public QAbstractListModel
{
    Q_OBJECT
public:
    explicit ChannelListModel(int displayflags, QObject *parent = 0);

    enum DisplayFlags {
        AnalogIn = 0x1,
        AnalogOut = 0x10
    };

    QVariant data(const QModelIndex & index, int role = Qt::DisplayRole) const;
    int rowCount(const QModelIndex & parent = QModelIndex()) const;
    Qt::ItemFlags flags(const QModelIndex &index) const;

signals:

public slots:

private:
    int displayflags;

    bool displayChannel(daq_channel *c) const;

};

#endif // CHANNELLISTMODEL_H
