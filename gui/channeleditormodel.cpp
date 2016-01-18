/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-04

--------------------------------------------------------------------------*/
#include "channeleditormodel.h"
#include "softrtdaq.h"
#include "rt.h"
#include "devicereferencemodel.h"
#include <comedilib.h>
#include <algorithm>
#include <iostream>

ChannelEditorModel::ChannelEditorModel(QObject *parent) :
    QAbstractListModel(parent)
{
}

void ChannelEditorModel::setRelatedModels(QAbstractListModel *channelModel, QAbstractListModel *rangeModel, QAbstractListModel *referenceModel)
{
    m_chan = channelModel;
    m_range = rangeModel;
    m_aref = referenceModel;
}

QVariant ChannelEditorModel::data(const QModelIndex & index, int role) const {
    daq_channel *c = channel(index.row());
    if ( role == Qt::EditRole ) {
        std::vector<daq_channel *>::iterator it;
        switch( index.column() ) {
        case Name:
            return QVariant(c->name);
        case Device:
            return QVariant(c->deviceno);
        case Type:
            return QVariant(subdevice_index(c->type));
        case Channel:
            return QVariant(c->channel);
        case Range:
            return QVariant(c->range);
        case Reference:
            return QVariant(DeviceReferenceModel::indexFromAref(c->aref));
        case ConversionFactor:
            return QVariant(c->gain);
        case Offset:
            return QVariant(c->offset);
        case ReadOffsetSource:
            if ( (it = std::find(config->io.channels.begin(), config->io.channels.end(), c->read_offset_src)) == config->io.channels.end() )
                return QVariant(-1);
            else
                return QVariant((int)(it - config->io.channels.begin()));
        case ReadOffsetLater:
            return QVariant((bool)c->read_offset_later);
        default:
            return QVariant();
        }
    }
    return QVariant();
}

bool ChannelEditorModel::sanitise(const QModelIndex &index) {
    daq_channel *c = channel(index.row());
    DeviceReferenceModel * ref_ = qobject_cast<DeviceReferenceModel *>(m_aref);
    switch ( index.column() ) {
    case Device:
        if ( daq_open_device(c->deviceno, &c->device) )
            return false;
        if ( rtdo_update_channel(c->handle) )
            return false;

    case Type:
        if ( -1 == (c->subdevice = comedi_find_subdevice_by_type(c->device, c->type, 0)) )
            return false;
        emit deviceChanged();

        if ( c->channel >= m_chan->rowCount() )
            c->channel = 0;

        if ( !ref_->hasAref(c->aref) )
            c->aref = ref_->getValidAref();

    case Channel:
        emit channelChanged();

        if ( c->range > m_range->rowCount() )
            c->range = 0;

    case Range:
        daq_create_converter(c);

    default:
        return true;
    }
}

bool ChannelEditorModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    daq_channel *c = channel(index.row());
    bool ret = true;
    int lastChanged = index.column();
    switch( index.column() ) {
    case Name:
        daq_set_channel_name(c, value.toString().toStdString().c_str());
        break;
    case Device:
        c->deviceno = value.toInt();
        ret = sanitise(index);
        lastChanged = Reference;
        break;
    case Type:
        c->type = subdevice_type(value.toInt());
        ret = sanitise(index);
        lastChanged = Reference;
        break;
    case Channel:
        c->channel = value.toInt();
        ret = sanitise(index);
        lastChanged = Reference;
        break;
    case Range:
        c->range = value.toInt();
        ret = sanitise(index);
        break;
    case Reference:
        c->aref = DeviceReferenceModel::arefFromIndex(value.toInt());
        break;
    case ConversionFactor:
        c->gain = value.toDouble();
        break;
    case Offset:
        c->offset = value.toDouble();
        break;
    case ReadOffsetSource:
        c->read_offset_src = channel(value.toInt());
        break;
    case ReadOffsetLater:
        c->read_offset_later = value.toBool();
        break;
    default:
        return false;
    }
    emit dataChanged(index, this->index(index.row(), lastChanged));
    emit channelsUpdated();
    return ret;
}

int ChannelEditorModel::rowCount(const QModelIndex & parent) const {
    return config->io.channels.size();
}

bool ChannelEditorModel::insertRow(int row, const QModelIndex &parent) {
    int ret;
    daq_channel *chan = new daq_channel;
    daq_create_channel(chan);
    if ( daq_setup_channel(chan) ) {
        daq_delete_channel(chan);
        return false;
    }
    daq_set_channel_name(chan, "New channel");
    if ( (ret = rtdo_add_channel(chan, 10000)) ) {
        if ( ret == EINVAL || ret == ENOMEM || ret == EMLINK )
            std::cerr << "Error: Channel creation failed." << std::endl;
        else if ( ret == ENODEV )
            std::cerr << "Error: Failed to open device in realtime mode." << std::endl;
        daq_delete_channel(chan);
        return false;
    }

    beginInsertRows(parent, row, row+1);
    config->io.channels.insert(config->io.channels.begin() + row, 1, chan);
    endInsertRows();
    emit channelsUpdated();
    return true;
}

bool ChannelEditorModel::removeRow(int row, const QModelIndex &parent) {
    daq_channel *c = channel(row);
    beginRemoveRows(parent, row, row+1);
    config->io.channels.erase(config->io.channels.begin() + row);
    endRemoveRows();

    for ( std::vector<daq_channel *>::iterator it = config->io.channels.begin(); it != config->io.channels.end(); ++it ) {
        if ( (*it)->read_offset_src == c )
            (*it)->read_offset_src = 0;
    }
    rtdo_remove_channel(c->handle);
    daq_delete_channel(c);
    free(c);

    emit channelsUpdated();
    return true;
}

int ChannelEditorModel::subdevice_index(comedi_subdevice_type subdev)
{
    return subdev == COMEDI_SUBD_AO ? 1 : 0;
}

comedi_subdevice_type ChannelEditorModel::subdevice_type(int index)
{
    return index ? COMEDI_SUBD_AO : COMEDI_SUBD_AI;
}

int ChannelEditorModel::columnCount(const QModelIndex &) const
{
    return FieldEnd_ - FieldStart_;
}
