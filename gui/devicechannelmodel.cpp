/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-04

--------------------------------------------------------------------------*/
#include "devicechannelmodel.h"
#include <comedilib.h>
#include "types.h"

DeviceChannelModel::DeviceChannelModel(ChannelEditorModel *editor, QDataWidgetMapper *mapper, QObject *parent) :
    QAbstractListModel(parent),
    editor(editor),
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
    daq_channel *c = editor->channel(mapper->currentIndex());
    if ( c )
        return comedi_get_n_channels(c->device, c->subdevice);
    else
        return 0;
}
