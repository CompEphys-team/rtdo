/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-04

--------------------------------------------------------------------------*/
#include "devicerangemodel.h"
#include "types.h"
#include <comedilib.h>

DeviceRangeModel::DeviceRangeModel(ChannelEditorModel *editor, QDataWidgetMapper *mapper, QObject *parent) :
    QAbstractListModel(parent),
    editor(editor),
    mapper(mapper)
{}

QVariant DeviceRangeModel::data(const QModelIndex &index, int role) const
{
    daq_channel *c = editor->channel(mapper->currentIndex());
    if ( c && role == Qt::DisplayRole ) {
        comedi_range *r = comedi_get_range(c->device, c->subdevice, c->channel, index.row());
        if ( !r )
            return QVariant();
        QString unit;
        if ( r->unit == UNIT_volt ) unit = "V";
        else if ( r->unit == UNIT_mA ) unit = "mA";
        else unit = "X";
        return QString("[%1 %3, %2 %3]").arg(r->min).arg(r->max).arg(unit);
    }
    return QVariant();
}

int DeviceRangeModel::rowCount(const QModelIndex &parent) const
{
    daq_channel *c = editor->channel(mapper->currentIndex());
    return c ? comedi_get_n_ranges(c->device, c->subdevice, c->channel) : 0;
}
