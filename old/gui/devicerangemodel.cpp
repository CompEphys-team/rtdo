/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-04

--------------------------------------------------------------------------*/
#include "devicerangemodel.h"
#include "config.h"

DeviceRangeModel::DeviceRangeModel(QDataWidgetMapper *mapper, QObject *parent) :
    QAbstractListModel(parent),
    mapper(mapper)
{}

QVariant DeviceRangeModel::data(const QModelIndex &index, int role) const
{
    if ( role == Qt::DisplayRole && mapper->currentIndex() >= 0 ) {
        Channel &c = config->io.channels.at(mapper->currentIndex());
        if ( c.hasRange(index.row()) ) {
            return QString("[%1 %3, %2 %3]")
                    .arg(c.rangeMin(index.row()))
                    .arg(c.rangeMax(index.row()))
                    .arg(QString::fromStdString(c.rangeUnit(index.row())));
        }
    }
    return QVariant();
}

int DeviceRangeModel::rowCount(const QModelIndex &parent) const
{
    if ( mapper->currentIndex() >= 0 ) {
        Channel &c = config->io.channels.at(mapper->currentIndex());
        unsigned int i = 0;
        while ( c.hasRange(i) )
            ++i;
        return (int) i;
    }
    return 0;
}
