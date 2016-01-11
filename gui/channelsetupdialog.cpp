/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-04

--------------------------------------------------------------------------*/
#include "channelsetupdialog.h"
#include "ui_channelsetupdialog.h"
#include "softrtdaq.h"
#include "rt.h"
#include "globals.h"
#include <comedilib.h>

ChannelSetupDialog::ChannelSetupDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ChannelSetupDialog),
    editor(new ChannelEditorModel),
    mapper(new QDataWidgetMapper),
    channelModel(new DeviceChannelModel(editor, mapper)),
    rangeModel(new DeviceRangeModel(editor, mapper)),
    arefModel(new DeviceReferenceModel(editor, mapper)),
    chanList(new ChannelListModel(ChannelListModel::AnalogIn | ChannelListModel::AnalogOut)),
    offsetSources(new ChannelListModel(ChannelListModel::AnalogIn))
{
    ui->setupUi(this);
    ui->channel->setModel(channelModel);
    ui->range->setModel(rangeModel);
    ui->reference->setModel(arefModel);

    editor->setRelatedModels(channelModel, rangeModel, arefModel);
    ui->readOffsetSource->setModel(offsetSources);
    ui->channelList->setModel(chanList);
    mapper->setModel(editor);
    mapper->setItemDelegate(new ComboboxDataDelegate(this));
    mapper->addMapping(ui->name, ChannelEditorModel::Name);
    mapper->addMapping(ui->device, ChannelEditorModel::Device);
    mapper->addMapping(ui->type, ChannelEditorModel::Type, "currentIndex");
    mapper->addMapping(ui->channel, ChannelEditorModel::Channel, "currentIndex");
    mapper->addMapping(ui->range, ChannelEditorModel::Range, "currentIndex");
    mapper->addMapping(ui->reference, ChannelEditorModel::Reference, "currentIndex");
    mapper->addMapping(ui->conversionFactor, ChannelEditorModel::ConversionFactor);
    mapper->addMapping(ui->offset, ChannelEditorModel::Offset);
    mapper->addMapping(ui->readOffsetSource, ChannelEditorModel::ReadOffsetSource, "currentIndex");
    mapper->addMapping(ui->readOffsetLater, ChannelEditorModel::ReadOffsetLater);
    mapper->setSubmitPolicy(QDataWidgetMapper::AutoSubmit);
    QItemSelectionModel *sm = ui->channelList->selectionModel();
    connect(sm, SIGNAL(currentRowChanged(QModelIndex, QModelIndex)),
            this, SLOT(selectionChanged(QModelIndex, QModelIndex)));
    connect(editor, SIGNAL(deviceChanged()),
            channelModel, SIGNAL(modelReset()),
            Qt::DirectConnection);
    connect(editor, SIGNAL(channelChanged()),
            rangeModel, SIGNAL(modelReset()),
            Qt::DirectConnection);
    connect(editor, SIGNAL(deviceChanged()),
            arefModel, SIGNAL(modelReset()),
            Qt::DirectConnection);
    connect(editor, SIGNAL(channelsUpdated()),
            this, SIGNAL(channelsUpdated()));
    connect(this, SIGNAL(channelsUpdated()),
            chanList, SIGNAL(modelReset()));
    connect(this, SIGNAL(channelsUpdated()),
            offsetSources, SIGNAL(modelReset()));
}

ChannelSetupDialog::~ChannelSetupDialog()
{
    delete ui;
    delete editor;
    delete mapper;
    delete channelModel;
    delete rangeModel;
    delete arefModel;
}

#include <iostream>
void ChannelSetupDialog::open()
{
    comedi_t *dev;
    ui->device->clear();
    for ( uint i = 0; i < DO_MAX_DEVICES; i++ ) {
        if ( !daq_open_device(i, &dev) ) {
            ui->device->addItem(QString("%1 (dev %2)").arg(QString(comedi_get_board_name(dev))).arg(i),
                                QVariant(i));
        }
    }

    QDialog::open();

    ui->channelList->reset();
}

void ChannelSetupDialog::addChannel()
{
    int pos = chanList->rowCount();
    editor->insertRow(pos);
    ui->channelList->setCurrentIndex(chanList->index(pos));
}

void ChannelSetupDialog::removeChannel()
{
    int pos = ui->channelList->currentIndex().row();
    if ( pos < 0 ) {
        ui->channelList->setCurrentIndex(chanList->index(0));
        return;
    }
    editor->removeRow(pos);
    ui->channelList->setCurrentIndex(chanList->index(pos >= chanList->rowCount() ? pos-1 : pos));
}

void ChannelSetupDialog::on_readOffsetNow_clicked()
{
    int err=0;
    daq_channel *src = config.io.channels.at(ui->readOffsetSource->currentIndex());
    double val = rtdo_read_now(src->handle, &err);
    if ( !err )
        editor->setData(editor->index(mapper->currentIndex(), ChannelEditorModel::Offset), QVariant(val), Qt::EditRole);
}

void ChannelSetupDialog::selectionChanged(QModelIndex index, QModelIndex previous)
{
    mapper->setCurrentIndex(index.row());
}


ComboboxDataDelegate::ComboboxDataDelegate(QObject *parent) : QItemDelegate(parent) {}

void ComboboxDataDelegate::setEditorData(QWidget *editor, const QModelIndex &index) const {
    QComboBox *device;
    if ( index.column() == ChannelEditorModel::Device && (device = qobject_cast<QComboBox *>(editor)) ) {
        device->setCurrentIndex(device->findData(index.data(Qt::EditRole)));
    } else {
        QItemDelegate::setEditorData(editor, index);
    }
}

void ComboboxDataDelegate::setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const {
    QComboBox *device;
    if ( index.column() == ChannelEditorModel::Device && (device = qobject_cast<QComboBox *>(editor)) ) {
        model->setData(index, device->currentData());
    } else {
        QItemDelegate::setModelData(editor, model, index);
    }
}
