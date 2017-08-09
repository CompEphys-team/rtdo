#include "daqdialog.h"
#include "ui_daqdialog.h"
#include <comedilib.h>
#include <QMessageBox>
#include <QStandardItemModel>

static std::vector<FilterMethod> methods;

DAQDialog::DAQDialog(Session &s, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::DAQDialog),
    session(s)
{
    ui->setupUi(this);
    ui->buttonBox->addButton("Apply and set as project default", QDialogButtonBox::ApplyRole);
    ui->samplesPerDt->setSuffix(ui->samplesPerDt->suffix().arg(session.project.dt()));
    chanUI[0] = {ui->channel, ui->range, ui->reference, ui->conversionFactor, ui->offset}; // Voltage in
    chanUI[1] = {ui->channel_2, ui->range_2, ui->reference_2, ui->conversionFactor_2, ui->offset_2}; // Current in
    chanUI[2] = {ui->channel_3, ui->range_3, ui->reference_3, ui->conversionFactor_3, ui->offset_3}; // Voltage cmd

    connect(ui->deviceNumber, SIGNAL(valueChanged(QString)), this, SLOT(updateChannelCapabilities()));
    for ( int i = 0; i < 3; i++ ) {
        connect(chanUI[i].channel, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int){
            updateChannelCapabilities(i, false);
        });
    }

    connect(&session, SIGNAL(DAQDataChanged()), this, SLOT(importData()));
    connect(this, SIGNAL(apply(DAQData)), &session, SLOT(setDAQData(DAQData)));

    methods = {FilterMethod::MovingAverage, FilterMethod::SavitzkyGolay23, FilterMethod::SavitzkyGolay45};
    QStringList labels;
    for ( FilterMethod m : methods )
        labels << QString::fromStdString(toString(m));
    ui->filterMethod->addItems(labels);

    importData();
}

DAQDialog::~DAQDialog()
{
    delete ui;
}

void DAQDialog::importData()
{
    const DAQData &p = session.daqData();

    ui->analogDAQ->setChecked(!p.simulate);
    ui->deviceNumber->setValue(p.devNo);
    ui->throttle->setValue(p.throttle);

    updateChannelCapabilities(-1, false);

    const ChnData *cp[] = {&p.voltageChn, &p.currentChn, &p.stimChn};
    for ( int i = 0; i < 3; i++ ) {
        chanUI[i].channel->setCurrentIndex(cp[i]->idx);
        chanUI[i].range->setCurrentIndex(cp[i]->range);
        chanUI[i].aref->setCurrentIndex(cp[i]->aref);
        chanUI[i].factor->setValue(cp[i]->gain);
        chanUI[i].offset->setValue(cp[i]->offset);
    }

    ui->cache->setChecked(p.cache.active);
    ui->numTraces->setValue(p.cache.numTraces);
    ui->average->setChecked(p.cache.averageWhileCollecting);
    ui->useMedian->setChecked(p.cache.useMedian);

    ui->filter->setChecked(p.filter.active);
    ui->samplesPerDt->setValue(p.filter.samplesPerDt);
    ui->filterWidth->setValue(p.filter.width);
    ui->filterMethod->setCurrentText(QString::fromStdString(toString(p.filter.method)));
}

DAQData DAQDialog::exportData()
{
    DAQData p;

    p.simulate = !ui->analogDAQ->isChecked();
    p.devNo = ui->deviceNumber->value();
    p.throttle = ui->throttle->value();

    ChnData *cp[] = {&p.voltageChn, &p.currentChn, &p.stimChn};
    for ( int i = 0; i < 3; i++ ) {
        int idx = chanUI[i].channel->currentIndex();
        cp[i]->idx = idx < 0 ? 0 : idx;
        cp[i]->active = idx >= 0;
        cp[i]->range = chanUI[i].range->currentIndex();
        cp[i]->aref = chanUI[i].aref->currentIndex();
        cp[i]->gain = chanUI[i].factor->value();
        cp[i]->offset = chanUI[i].offset->value();
    }

    p.cache.active = ui->cache->isChecked();
    p.cache.numTraces = ui->numTraces->value();
    p.cache.averageWhileCollecting = ui->average->isChecked();
    p.cache.useMedian = ui->useMedian->isChecked();

    p.filter.active = ui->filter->isChecked();
    p.filter.samplesPerDt = ui->samplesPerDt->value();
    p.filter.width = ui->filterWidth->value();
    if ( p.filter.width % 2 == 0 )
        ui->filterWidth->setValue(++p.filter.width);
    for ( FilterMethod m : methods ) {
        if ( QString::fromStdString(toString(m)) == ui->filterMethod->currentText() ) {
            p.filter.method = m;
            break;
        }
    }

    emit apply(p);
    return p;
}

void DAQDialog::updateChannelCapabilities(int tab, bool checkDevice)
{
    DAQData p;
    p.devNo = ui->deviceNumber->value();
    comedi_t *dev = comedi_open(p.devname().c_str());
    if ( dev ) {
        ui->deviceName->setText(comedi_get_board_name(dev));
        int aidev = comedi_find_subdevice_by_type(dev, COMEDI_SUBD_AI, 0);
        int aodev = comedi_find_subdevice_by_type(dev, COMEDI_SUBD_AO, 0);
        int subdev[3] = {aidev, aidev, aodev};
        if ( tab < 0 )
            for ( int i = 0; i < 3; i++ )
                updateSingleChannelCapabilities(dev, subdev[i], chanUI[i]);
        else
            updateSingleChannelCapabilities(dev, subdev[tab], chanUI[tab]);
    } else if ( checkDevice && isVisible() && ui->analogDAQ->isChecked() ) {
        ui->deviceName->setText("No such device");
        QMessageBox err;
        err.setText(QString("Unable to find or open device \"%1\"\n%2")
                    .arg(QString::fromStdString(p.devname()))
                    .arg(comedi_strerror(comedi_errno())));
        err.exec();
    }
}

void DAQDialog::updateSingleChannelCapabilities(void *vdev, int subdev, DAQDialog::ChannelUI &cui)
{
    comedi_t *dev = (comedi_t*)vdev;
    int currentChannel = cui.channel->currentIndex();

    int nChannels = comedi_get_n_channels(dev, subdev);
    if ( nChannels != cui.channel->count() ) {
        QStringList labels;
        for ( int i = 0; i < nChannels; i++ ) {
            labels << QString::number(i);
        }
        cui.channel->clear();
        cui.channel->addItems(labels);
        currentChannel = currentChannel < 0 || currentChannel > labels.size() ? 0 : currentChannel;
        cui.channel->setCurrentIndex(currentChannel);
    }

    int nRanges = comedi_get_n_ranges(dev, subdev, currentChannel);
    {
        QStringList labels;
        for ( int i = 0; i < nRanges; i++ ) {
            comedi_range *range = comedi_get_range(dev, subdev, currentChannel, i);
            QString unit;
            if ( range->unit == UNIT_mA ) unit = "mA";
            else if ( range->unit == UNIT_volt ) unit = "V";
            labels << QString("%1 to %2 %3").arg(range->min).arg(range->max).arg(unit);
        }
        int currentIdx = cui.range->currentIndex();
        cui.range->clear();
        cui.range->addItems(labels);
        cui.range->setCurrentIndex(currentIdx < 0 || currentIdx > labels.size() ? 0 : currentIdx);
    }

    unsigned int sdf = comedi_get_subdevice_flags(dev, subdev);
    {
        unsigned int ref[4] = {SDF_GROUND, SDF_COMMON, SDF_DIFF, SDF_OTHER};
        QStandardItemModel *model = qobject_cast<QStandardItemModel*>(cui.aref->model());
        for ( int i = 0; i < 4; i++ ) {
            QStandardItem *item = model->item(i);
            item->setFlags((sdf & ref[i]) ? (item->flags() | Qt::ItemIsEnabled) : (item->flags() & ~Qt::ItemIsEnabled));
        }
    }
}

void DAQDialog::on_buttonBox_clicked(QAbstractButton *button)
{
    QDialogButtonBox::ButtonRole role = ui->buttonBox->buttonRole(button);
    if ( role == QDialogButtonBox::AcceptRole ) { // OK
        exportData();
        close();
    } else if ( role == QDialogButtonBox::ApplyRole ) { // Apply & set project default
        session.project.setDaqData(exportData());
        close();
    } else { // Cancel
        importData();
        close();
    }
}
