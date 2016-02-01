/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-04

--------------------------------------------------------------------------*/
#include "devicechannelmodel.h"
#include "config.h"

DeviceChannelModel::DeviceChannelModel(QDataWidgetMapper *mapper, QObject *parent) :
    QAbstractListModel(parent),
    mapper(mapper)
{}

QVariant DeviceChannelModel::data(const QModelIndex &index, int role) const
{
    if ( role == Qt::DisplayRole )
        return QString::number(index.row());
    return QVariant();
}

int DeviceChannelModel::rowCount(const QModelIndex &parent) const
{
    if ( mapper->currentIndex() >= 0 )
        return config->io.channels.at(mapper->currentIndex()).numChannels();
    return 0;
}
