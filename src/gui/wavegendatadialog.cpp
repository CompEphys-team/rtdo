#include "wavegendatadialog.h"
#include "ui_wavegendatadialog.h"

WavegenDataDialog::WavegenDataDialog(Session &s, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::WavegenDataDialog),
    session(s)
{
    ui->setupUi(this);
    connect(this, SIGNAL(accepted()), this, SLOT(exportData()));
    connect(this, SIGNAL(rejected()), this, SLOT(importData()));
    connect(&session, &Session::actionLogged, [=](QString actor, QString action, QString, int) {
        if ( actor == "Config" && action == "cfg" )
            importData();
    });
    connect(this, SIGNAL(apply(WavegenData)), &session, SLOT(setWavegenData(WavegenData)));

    importData();
}

WavegenDataDialog::~WavegenDataDialog()
{
    delete ui;
}

void WavegenDataDialog::importData()
{
    const WavegenData &p = session.wavegenData();
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
    scalar maxDeviation = session.stimulationData().maxVoltage-session.stimulationData().baseV > session.stimulationData().baseV-session.stimulationData().minVoltage
            ? session.stimulationData().maxVoltage - session.stimulationData().baseV
            : session.stimulationData().baseV - session.stimulationData().minVoltage;
    if ( ui->nBinsBubbleTime->value() > 0 )
        dims.push_back(MAPEDimension {MAPEDimension::Func::BestBubbleTime, 0, session.stimulationData().duration, (size_t)ui->nBinsBubbleTime->value()});
    if ( ui->nBinsBubbleDuration->value() > 0 )
        dims.push_back(MAPEDimension {MAPEDimension::Func::BestBubbleDuration, 0, session.stimulationData().duration, (size_t)ui->nBinsBubbleDuration->value()});
    if ( ui->nBinsVoltageDeviation->value() > 0 )
        dims.push_back(MAPEDimension {MAPEDimension::Func::VoltageDeviation, 0, maxDeviation, (size_t)ui->nBinsVoltageDeviation->value()});
    if ( ui->nBinsVoltageIntegral->value() > 0 )
        dims.push_back(MAPEDimension {MAPEDimension::Func::VoltageIntegral, 0, maxDeviation * session.stimulationData().duration, (size_t)ui->nBinsVoltageIntegral->value()});
    p.mapeDimensions = dims;

    emit apply(p);
}
