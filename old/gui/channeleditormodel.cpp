/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-04

--------------------------------------------------------------------------*/
#include "channeleditormodel.h"
#include "devicereferencemodel.h"
#include "channel.h"

ChannelEditorModel::ChannelEditorModel(QObject *parent) :
    QAbstractListModel(parent)
{
}

QVariant ChannelEditorModel::data(const QModelIndex & index, int role) const {
    if ( role == Qt::EditRole ) {
        if ( index.row() < 0 )
            return QVariant();
        Channel &c = config->io.channels.at(index.row());
        int i = 0;
        switch( index.column() ) {
        case Name:
            return QVariant(QString::fromStdString(c.name()));
        case Device:
            return QVariant(c.device());
        case Type:
            return QVariant(c.direction());
        case ChannelField:
            return QVariant(c.channel());
        case Range:
            return QVariant(c.range());
        case Reference:
            return QVariant(c.aref());
        case ConversionFactor:
            return QVariant(c.conversionFactor());
        case Offset:
            return QVariant(c.offset());
        case ReadOffsetSource:
            for ( Channel &search : config->io.channels ) {
                if ( search.ID() == c.offsetSource() )
                    return QVariant(i+1);
                ++i;
            }
            return QVariant(0);
        case ReadResetButton:
            if ( c.direction() == Channel::AnalogIn ) {
                return QVariant("Read current value");
            } else {
                return QVariant("Reset output");
            }
        default:
            return QVariant();
        }
    }
    return QVariant();
}

bool ChannelEditorModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    Channel &c = config->io.channels.at(index.row());
    switch( index.column() ) {
    case Name:
        c.setName(value.toString().toStdString());
        break;
    case Device:
        if ( !c.setDevice(value.toInt()) )
            return false;
        emit dataChanged(index, this->index(index.row(), Reference));
        break;
    case Type:
        if ( !c.setDirection((Channel::Direction)value.toInt()) )
            return false;
        emit dataChanged(index, this->index(index.row(), Reference));
        break;
    case ChannelField:
        if ( !c.setChannel(value.toInt()) )
            return false;
        emit dataChanged(index, this->index(index.row(), Reference));
        break;
    case Range:
        if ( !c.setRange(value.toInt()) )
            return false;
        break;
    case Reference:
        if ( !c.setAref((Channel::Aref)value.toInt()) )
            return false;
        break;
    case ConversionFactor:
        c.setConversionFactor(value.toDouble());
        break;
    case Offset:
        c.setOffset(value.toDouble());
        break;
    case ReadOffsetSource:
        if ( value.toInt() > 0 )
            c.setOffsetSource(config->io.channels.at(value.toInt()-1).ID());
        else
            c.setOffsetSource(0);
        break;
    default:
        return false;
    }
    emit channelsUpdated();
    return true;
}

int ChannelEditorModel::rowCount(const QModelIndex & parent) const {
    return config->io.channels.size();
}

bool ChannelEditorModel::insertRow(int row, const QModelIndex &parent) {
    beginInsertRows(parent, row, row+1);
    config->io.channels.insert(config->io.channels.begin() + row, 1, Channel(Channel::AnalogIn));
    endInsertRows();
    emit channelsUpdated();
    return true;
}

bool ChannelEditorModel::removeRow(int row, const QModelIndex &parent) {
    beginRemoveRows(parent, row, row+1);
    config->io.channels.erase(config->io.channels.begin() + row);
    endRemoveRows();
    emit channelsUpdated();
    return true;
}

void ChannelEditorModel::read_reset(int index, double &sample)
{
    Channel &c = config->io.channels.at(index);
    c.readOffset();
    QModelIndex idx = this->index(index, Offset);
    emit dataChanged(idx, idx);
    if ( c.direction() == Channel::AnalogIn ) {
        c.read(sample, true);
    } else {
        c.write(sample);
    }
}

int ChannelEditorModel::columnCount(const QModelIndex &) const
{
    return FieldEnd_ - FieldStart_;
}
