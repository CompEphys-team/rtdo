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
#include "softrtdaq.h"
#include "globals.h"
#include "config.h"

class ChannelEditorModel : public QAbstractListModel
{
    Q_OBJECT
public:
    explicit ChannelEditorModel(QObject *parent = 0);

    void setRelatedModels(QAbstractListModel *channelModel, QAbstractListModel *rangeModel, QAbstractListModel *referenceModel);

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
        Channel,
        Range,
        Reference,
        ConversionFactor,
        Offset,
        ReadOffsetSource,
        ReadOffsetLater,
        FieldEnd_
    };

    static int subdevice_index(comedi_subdevice_type subdev);
    static comedi_subdevice_type subdevice_type(int index);

    inline daq_channel *channel(int index) const { return index >= 0 ? config->io.channels[index] : 0; }

signals:
    void deviceChanged();
    void channelChanged();
    void channelsUpdated();

private:
    int columnCount(const QModelIndex &parent) const;
    bool sanitise(const QModelIndex &index);

    daq_channel *c;
    QAbstractListModel *m_chan;
    QAbstractListModel *m_range;
    QAbstractListModel *m_aref;

};

#endif // CHANNELEDITORMODEL_H
