/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-04

--------------------------------------------------------------------------*/
#include "channellistmodel.h"
#include "globals.h"
#include "config.h"

ChannelListModel::ChannelListModel(int displayflags, QObject *parent) :
    QAbstractListModel(parent),
    displayflags(displayflags)
{
}

QVariant ChannelListModel::data(const QModelIndex &index, int role) const
{
    daq_channel *c;
    try {
        c = config->io.channels.at(index.row());
    } catch (...) {
        return QVariant();
    }
    if ( c && role == Qt::DisplayRole ) {
        return QString("%1 (dev %2, %3 %4)")
                .arg(QString(c->name))
                .arg(c->deviceno)
                .arg(QString(c->type == COMEDI_SUBD_AO ? "out" : "in"))
                .arg(c->channel);
    }
    return QVariant();
}

int ChannelListModel::rowCount(const QModelIndex &parent) const
{
    return config->io.channels.size();
}

Qt::ItemFlags ChannelListModel::flags(const QModelIndex &index) const
{
    daq_channel *c = config->io.channels.at(index.row());
    if ( c && displayChannel(c) )
        return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
    else
        return Qt::NoItemFlags;
}

bool ChannelListModel::displayChannel(daq_channel *c) const
{
    return c && (
            (displayflags & AnalogIn && c->type == COMEDI_SUBD_AI) ||
            (displayflags & AnalogOut && c->type == COMEDI_SUBD_AO)
                );
}
