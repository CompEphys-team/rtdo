/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-04

--------------------------------------------------------------------------*/
#include "devicereferencemodel.h"
#include <comedilib.h>

#define NREF 4
int refmap[NREF] = {
    AREF_GROUND,
    AREF_COMMON,
    AREF_DIFF,
    AREF_OTHER
};
int flagmap[NREF] = {
    SDF_GROUND,
    SDF_COMMON,
    SDF_DIFF,
    SDF_OTHER
};
QString refnames[NREF] = {
    "Ground",
    "Common",
    "Diff",
    "Other"
};

DeviceReferenceModel::DeviceReferenceModel(ChannelEditorModel *editor, QDataWidgetMapper *mapper, QObject *parent) :
    QAbstractListModel(parent),
    editor(editor),
    mapper(mapper)
{
}

QVariant DeviceReferenceModel::data(const QModelIndex &index, int role) const
{
    if ( role == Qt::DisplayRole ) {
        return refnames[index.row()];
    }
    return QVariant();
}

int DeviceReferenceModel::rowCount(const QModelIndex &parent) const
{
    return NREF;
}

Qt::ItemFlags DeviceReferenceModel::flags(const QModelIndex &index) const
{
    daq_channel *c = editor->channel(mapper->currentIndex());
    if ( c && (comedi_get_subdevice_flags(c->device, c->subdevice) & flagmap[index.row()]) )
        return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
    else
        return Qt::NoItemFlags;
}

int DeviceReferenceModel::indexFromAref(int aref)
{
    for (int i = 0; i < NREF; i++ )
        if ( refmap[i] == aref )
            return i;
    return -1;
}

int DeviceReferenceModel::arefFromIndex(int index)
{
    return refmap[index];
}

bool DeviceReferenceModel::hasAref(int aref)
{
    daq_channel *c = editor->channel(mapper->currentIndex());
    return c && (comedi_get_subdevice_flags(c->device, c->subdevice) & flagmap[indexFromAref(aref)]);
}

int DeviceReferenceModel::getValidAref()
{
    daq_channel *c = editor->channel(mapper->currentIndex());
    if ( !c )
        return -1;
    int sdflags = comedi_get_subdevice_flags(c->device, c->subdevice);
    for ( int i = 0; i < NREF; i++ )
        if ( sdflags & flagmap[i] )
            return arefFromIndex(i);
    return -1;
}
