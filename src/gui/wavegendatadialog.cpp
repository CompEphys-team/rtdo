#include "wavegendatadialog.h"
#include "ui_wavegendatadialog.h"

WavegenDataDialog::WavegenDataDialog(Session &s, int historicIndex, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::WavegenDataDialog),
    session(s),
    historicIndex(historicIndex)
{
    ui->setupUi(this);
    setAttribute(Qt::WA_DeleteOnClose);

    if ( historicIndex < 0 ) {
        connect(&session, SIGNAL(wavegenDataChanged()), this, SLOT(importData()));
        connect(this, SIGNAL(apply(WavegenData)), &session, SLOT(setWavegenData(WavegenData)));
    } else {
        ui->buttonBox->setStandardButtons(QDialogButtonBox::Close);
    }

    importData();
}

WavegenDataDialog::~WavegenDataDialog()
{
    delete ui;
}

void WavegenDataDialog::importData()
{
    WavegenData p = historicIndex < 0 ? session.qWavegenData() : session.wavegenData(historicIndex);
    ui->numSigmaAdjustWaveforms->setValue(p.numSigmaAdjustWaveforms);
    ui->nInitialWaves->setValue(p.nInitialWaves);
    ui->nGroupsPerWave->setValue(p.nGroupsPerWave);
    ui->nWavesPerEpoch->setValue(p.nWavesPerEpoch);
    ui->maxIterations->setValue(p.maxIterations);
    ui->useBaseParameters->setChecked(p.useBaseParameters);
    ui->rerandomiseParameters->setChecked(p.rerandomiseParameters);

    QStringList str;
    for ( size_t i : p.precisionIncreaseEpochs )
        str << QString::number(i);
    ui->precisionIncreaseEpochs->setText(str.join(','));

    ui->nBinsBubbleTime->setValue(0);
    ui->nBinsBubbleDuration->setValue(0);
    ui->nBinsVoltageDeviation->setValue(0);
    ui->nBinsVoltageIntegral->setValue(0);
    for ( const MAPEDimension &dim : p.mapeDimensions ) {
        if ( dim.func == MAPEDimension::Func::BestBubbleTime )
            ui->nBinsBubbleTime->setValue(dim.resolution);
        if ( dim.func == MAPEDimension::Func::BestBubbleDuration)
            ui->nBinsBubbleDuration->setValue(dim.resolution);
        if ( dim.func == MAPEDimension::Func::VoltageDeviation)
            ui->nBinsVoltageDeviation->setValue(dim.resolution);
        if ( dim.func == MAPEDimension::Func::VoltageIntegral)
            ui->nBinsVoltageIntegral->setValue(dim.resolution);
    }

    ui->dt->setValue(p.dt);
    ui->noise_sd->setValue(p.noise_sd);
}

void WavegenDataDialog::exportData()
{
    WavegenData p;
    p.numSigmaAdjustWaveforms = ui->numSigmaAdjustWaveforms->value();
    p.nInitialWaves = ui->nInitialWaves->value();
    p.nGroupsPerWave = ui->nGroupsPerWave->value();
    p.nWavesPerEpoch = ui->nWavesPerEpoch->value();
    p.maxIterations = ui->maxIterations->value();
    p.useBaseParameters = ui->useBaseParameters->isChecked();
    p.rerandomiseParameters = ui->rerandomiseParameters->isChecked();

    std::vector<size_t> prec;
    for ( QString n : ui->precisionIncreaseEpochs->text().split(',', QString::SkipEmptyParts) ) {
        bool ok;
        int num = n.toInt(&ok);
        if ( ok )
            prec.push_back(num);
    }
    std::sort(prec.begin(), prec.end());
    p.precisionIncreaseEpochs = prec;

    std::vector<MAPEDimension> dims;
    const StimulationData &stimd = session.qStimulationData();
    scalar maxDeviation = stimd.maxVoltage-stimd.baseV > stimd.baseV-stimd.minVoltage
            ? stimd.maxVoltage - stimd.baseV
            : stimd.baseV - stimd.minVoltage;
    if ( ui->nBinsBubbleTime->value() > 0 )
        dims.push_back(MAPEDimension {MAPEDimension::Func::BestBubbleTime, 0, stimd.duration, (size_t)ui->nBinsBubbleTime->value()});
    if ( ui->nBinsBubbleDuration->value() > 0 )
        dims.push_back(MAPEDimension {MAPEDimension::Func::BestBubbleDuration, 0, stimd.duration, (size_t)ui->nBinsBubbleDuration->value()});
    if ( ui->nBinsVoltageDeviation->value() > 0 )
        dims.push_back(MAPEDimension {MAPEDimension::Func::VoltageDeviation, 0, maxDeviation, (size_t)ui->nBinsVoltageDeviation->value()});
    if ( ui->nBinsVoltageIntegral->value() > 0 )
        dims.push_back(MAPEDimension {MAPEDimension::Func::VoltageIntegral, 0, maxDeviation * stimd.duration, (size_t)ui->nBinsVoltageIntegral->value()});
    p.mapeDimensions = dims;

    p.dt = ui->dt->value();
    p.noise_sd = ui->noise_sd->value();

    emit apply(p);
}

void WavegenDataDialog::on_buttonBox_clicked(QAbstractButton *button)
{
    QDialogButtonBox::ButtonRole role = ui->buttonBox->buttonRole(button);
    if ( role  == QDialogButtonBox::AcceptRole ) {
        //ok
        exportData();
        close();
    } else if ( role == QDialogButtonBox::ApplyRole ) {
        // apply
        exportData();
    } else {
        // cancel
        importData();
        close();
    }
}
