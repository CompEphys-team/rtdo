/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-04

--------------------------------------------------------------------------*/
#include "channellistmodel.h"
#include "config.h"

ChannelListModel::ChannelListModel(int displayflags, QObject *parent) :
    QAbstractListModel(parent),
    displayflags(displayflags)
{
}

QVariant ChannelListModel::data(const QModelIndex &index, int role) const
{
    if ( role == Qt::DisplayRole && index.row() >= 0 ) {
        Channel &c = config->io.channels.at(index.row());
        QString type;
        switch ( c.type() ) {
        case Channel::AnalogIn:  type = "in";  break;
        case Channel::AnalogOut: type = "out"; break;
        case Channel::Simulator: type = "sim"; break;
        }
        return QString("%1 (dev %2, %3 %4)")
                .arg(QString::fromStdString(c.name()))
                .arg(c.device())
                .arg(type)
                .arg(c.channel());
    }
    return QVariant();
}

int ChannelListModel::rowCount(const QModelIndex &parent) const
{
    return config->io.channels.size();
}

Qt::ItemFlags ChannelListModel::flags(const QModelIndex &index) const
{
    Channel &c = config->io.channels.at(index.row());
    if ( (displayflags & AnalogIn && c.type() == Channel::AnalogIn) ||
         (displayflags & AnalogOut && c.type() == Channel::AnalogOut) ||
         (displayflags & (AnalogIn | AnalogOut) && c.type() == Channel::Simulator) )
        return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
    else
        return Qt::NoItemFlags;
}
