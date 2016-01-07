/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-04

--------------------------------------------------------------------------*/
#ifndef DEVICEREFERENCEMODEL_H
#define DEVICEREFERENCEMODEL_H

#include <QAbstractListModel>
#include <QDataWidgetMapper>
#include "channeleditormodel.h"

class DeviceReferenceModel : public QAbstractListModel
{
    Q_OBJECT
public:
    explicit DeviceReferenceModel(ChannelEditorModel *editor, QDataWidgetMapper *mapper, QObject *parent = 0);
    QVariant data(const QModelIndex &index, int role) const;
    int rowCount(const QModelIndex &parent = QModelIndex()) const;
    Qt::ItemFlags flags(const QModelIndex &index) const;

    static int indexFromAref(int aref);
    static int arefFromIndex(int index);

    bool hasAref(int aref);
    int getValidAref();

protected:
    ChannelEditorModel *editor;
    QDataWidgetMapper *mapper;
};

#endif // DEVICEREFERENCEMODEL_H
