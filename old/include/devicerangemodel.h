/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-04

--------------------------------------------------------------------------*/
#ifndef DEVICERANGEMODEL_H
#define DEVICERANGEMODEL_H

#include <QAbstractListModel>
#include <QDataWidgetMapper>

class DeviceRangeModel : public QAbstractListModel
{
    Q_OBJECT
public:
    explicit DeviceRangeModel(QDataWidgetMapper *mapper, QObject *parent = 0);
    QVariant data(const QModelIndex &index, int role) const;
    int rowCount(const QModelIndex &parent = QModelIndex()) const;

protected:
    QDataWidgetMapper *mapper;
};

#endif // DEVICERANGEMODEL_H
