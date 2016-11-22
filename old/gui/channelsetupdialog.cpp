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
#include "realtimeenvironment.h"

ChannelSetupDialog::ChannelSetupDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ChannelSetupDialog),
    editor(new ChannelEditorModel),
    mapper(new QDataWidgetMapper),
    channelModel(new DeviceChannelModel(mapper)),
    rangeModel(new DeviceRangeModel(mapper)),
    arefModel(new DeviceReferenceModel(mapper)),
    chanList(new ChannelListModel(ChannelListModel::AnalogIn | ChannelListModel::AnalogOut)),
    offsetSources(new ChannelListModel(ChannelListModel::AnalogIn | ChannelListModel::None))
{
    ui->setupUi(this);
    ui->channel->setModel(channelModel);
    ui->range->setModel(rangeModel);
    ui->reference->setModel(arefModel);
    ui->readOffsetSource->setModel(offsetSources);
    ui->channelList->setModel(chanList);

    mapper->setModel(editor);
    mapper->setItemDelegate(new ComboboxDataDelegate(this));
    mapper->addMapping(ui->name, ChannelEditorModel::Name);
    mapper->addMapping(ui->device, ChannelEditorModel::Device);
    mapper->addMapping(ui->type, ChannelEditorModel::Type, "currentIndex");
    mapper->addMapping(ui->channel, ChannelEditorModel::ChannelField, "currentIndex");
    mapper->addMapping(ui->range, ChannelEditorModel::Range, "currentIndex");
    mapper->addMapping(ui->reference, ChannelEditorModel::Reference, "currentIndex");
    mapper->addMapping(ui->conversionFactor, ChannelEditorModel::ConversionFactor);
    mapper->addMapping(ui->offset, ChannelEditorModel::Offset);
    mapper->addMapping(ui->readOffsetSource, ChannelEditorModel::ReadOffsetSource, "currentIndex");
    mapper->addMapping(ui->read_reset, ChannelEditorModel::ReadResetButton, "text");
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
    connect(ui->read_reset, &QPushButton::clicked,
            [=](){
                double sample = ui->read_reset_val->value();
                editor->read_reset(mapper->currentIndex(), sample);
                ui->read_reset_val->setValue(sample);
            });
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

void ChannelSetupDialog::open()
{
    RealtimeEnvironment* &env = RealtimeEnvironment::env();
    std::string name;
    for ( int i = 0; i < 32; ++i) { // Doubt there's anyone with more than 32 devices!
        try {
            name = env->getDeviceName(i);
        } catch ( RealtimeException & ) {
            continue;
        }
        ui->device->addItem(QString("%1 (dev %2)").arg(QString::fromStdString(name)).arg(i),
                            QVariant(i));
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

void ChannelSetupDialog::selectionChanged(QModelIndex index, QModelIndex previous)
{
    mapper->setCurrentIndex(index.row());
    ui->read_reset_val->setValue(0.0);
    ui->read_reset_val->clear();
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
