/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-04

--------------------------------------------------------------------------*/
#ifndef CHANNELEDITORMODEL_H
#define CHANNELEDITORMODEL_H

#include <QAbstractListModel>
#include "config.h"

class ChannelEditorModel : public QAbstractListModel
{
    Q_OBJECT
public:
    explicit ChannelEditorModel(QObject *parent = 0);

    QVariant data(const QModelIndex & index, int role = Qt::DisplayRole) const;
    bool setData(const QModelIndex &index, const QVariant &value, int role);
    int rowCount(const QModelIndex & parent = QModelIndex()) const;
    bool insertRow(int row, const QModelIndex &parent = QModelIndex());
    bool removeRow(int row, const QModelIndex &parent = QModelIndex());

    enum Field {
        FieldStart_,
        Name,
        Device,
        Type,
        ChannelField,
        Range,
        Reference,
        ConversionFactor,
        Offset,
        ReadOffsetSource,
        ReadResetButton,
        FieldEnd_
    };

signals:
    void deviceChanged();
    void channelChanged();
    void channelsUpdated();

public slots:
    void read_reset(int index, double &sample);

private:
    int columnCount(const QModelIndex &parent) const;

};

#endif // CHANNELEDITORMODEL_H
