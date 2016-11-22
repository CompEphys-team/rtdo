/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-04

--------------------------------------------------------------------------*/
#include "devicereferencemodel.h"
#include "config.h"

// Compare against Channel::Aref
QString refnames[] = {
    "Ground",
    "Common",
    "Diff",
    "Other"
};

DeviceReferenceModel::DeviceReferenceModel(QDataWidgetMapper *mapper, QObject *parent) :
    QAbstractListModel(parent),
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
    return sizeof(refnames) / sizeof(QString);
}

Qt::ItemFlags DeviceReferenceModel::flags(const QModelIndex &index) const
{
    if ( mapper->currentIndex() >= 0 && config->io.channels.at(mapper->currentIndex()).hasAref((Channel::Aref)index.row()) )
        return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
    else
        return Qt::NoItemFlags;
}
