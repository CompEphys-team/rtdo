#include "samplingprofiledialog.h"
#include "ui_samplingprofiledialog.h"
#include "session.h"
#include <QDoubleSpinBox>
#include "wavesource.h"

SamplingProfileDialog::SamplingProfileDialog(Session &s, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::SamplingProfileDialog),
    session(s)
{
    ui->setupUi(this);

    connect(this, SIGNAL(generate(SamplingProfiler::Profile)), &session.samplingProfiler(), SLOT(generate(SamplingProfiler::Profile)));

    connect(&session.wavesets(), SIGNAL(addedSet()), this, SLOT(updateCombo()));
    connect(&session.samplingProfiler(), SIGNAL(done()), this, SLOT(updatePresets()));

    QStringList labels;
    int rows = session.project.model().adjustableParams.size();
    ui->table->setRowCount(rows);
    mins.resize(rows);
    maxes.resize(rows);
    int i = 0;
    for ( AdjustableParam const& p : session.project.model().adjustableParams ) {
        labels << QString::fromStdString(p.name);

        QTableWidgetItem *unif = new QTableWidgetItem();
        unif->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled);
        unif->setCheckState(Qt::Checked);
        ui->table->setItem(i, 0, unif);

        int decimals = 4 - log10(fabs(p.min == 0 ? p.initial : p.min));
        QDoubleSpinBox *min = new QDoubleSpinBox();
        min->setDecimals(decimals);
        min->setRange(p.min, p.max);
        min->setValue(p.min);
        ui->table->setCellWidget(i, 1, min);
        mins[i] = min;

        QDoubleSpinBox *max = new QDoubleSpinBox();
        max->setDecimals(decimals);
        max->setRange(p.min, p.max);
        max->setValue(p.max);
        ui->table->setCellWidget(i, 2, max);
        maxes[i] = max;

        ++i;
    }
    ui->table->setVerticalHeaderLabels(labels);
    ui->table->setColumnWidth(0, 50);

    ui->target->addItems(labels);
    connect(ui->cbSelection, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int i){
        if ( i < 0 )
            return;
        WaveSource src = ui->cbSelection->currentData().value<WaveSource>();
        if ( src.archive() )
            ui->target->setCurrentIndex(src.archive()->param);
    });

    ui->interval->setValue(session.qRunData().simCycles);

    updateCombo();
    updatePresets();
}

SamplingProfileDialog::~SamplingProfileDialog()
{
    delete ui;
}

void SamplingProfileDialog::updateCombo()
{
    int currentIdx = ui->cbSelection->currentIndex();
    WaveSource currentData = ui->cbSelection->currentData().value<WaveSource>();
    ui->cbSelection->clear();
    std::vector<WaveSource> sources = session.wavesets().sources();
    for ( const WaveSource &src : sources ) {
        ui->cbSelection->addItem(src.prettyName(), QVariant::fromValue(src));
    }

    if ( currentIdx < 0 )
        return;

    ui->cbSelection->setCurrentIndex(currentData.index());
}

void SamplingProfileDialog::updatePresets()
{
    if ( ui->preset->count() == 0 ) {
        ui->preset->addItem("Full range");
        ui->preset->addItem("Close range (# sigmas)");
    }

    int currentIdx = ui->preset->currentIndex();
    while ( ui->preset->count() > nHardPresets )
        ui->preset->removeItem(nHardPresets);

    int i = 0;
    for ( SamplingProfiler::Profile const& prof : session.samplingProfiler().profiles() )
        ui->preset->addItem(QString("Profile %1, probing %2 on %3")
                            .arg(i++)
                            .arg(QString::fromStdString(session.project.model().adjustableParams[prof.target].name))
                            .arg(prof.src.prettyName()));

    ui->preset->setCurrentIndex(currentIdx);
}

void SamplingProfileDialog::setCloseRange(int i)
{
    double min, max;
    const AdjustableParam &p = session.project.model().adjustableParams[i];
    if ( p.multiplicative ) {
        max = p.initial * pow(1.0 + p.adjustedSigma, ui->presetSig->value());
        min = p.initial * pow(1.0 - p.adjustedSigma, ui->presetSig->value());
    } else {
        max = p.initial + p.adjustedSigma*ui->presetSig->value();
        min = p.initial - p.adjustedSigma*ui->presetSig->value();
    }
    if ( max > p.max )
        max = p.max;
    if ( min < p.min )
        min = p.min;
    maxes[i]->setValue(max);
    mins[i]->setValue(min);
}

void SamplingProfileDialog::on_btnPreset_clicked()
{
    if ( ui->preset->currentIndex() < 0 )
        return;
    int preset = ui->preset->currentIndex();

    if ( preset == 1 ) { // Close range
        for ( int i = 0; i < int(mins.size()); i++ ) {
            setCloseRange(i);
        }
    } else if ( preset == 0 ) { // Full range
        for ( int i = 0; i < int(mins.size()); i++ ) {
            mins[i]->setValue(mins[i]->minimum());
            maxes[i]->setValue(maxes[i]->maximum());
        }
    } else {
        const SamplingProfiler::Profile &prof = session.samplingProfiler().profiles().at(preset - nHardPresets);
        for ( int i = 0; i < prof.uniform.size(); i++ ) {
            mins[i]->setValue(prof.value1[i]);
            maxes[i]->setValue(prof.value2[i]);
            ui->table->item(i, 0)->setCheckState(prof.uniform[i] ? Qt::Checked : Qt::Unchecked);
        }
    }
}

void SamplingProfileDialog::on_btnStart_clicked()
{
    if ( ui->cbSelection->currentIndex() < 0 )
        return;

    SamplingProfiler::Profile prof(ui->cbSelection->currentData().value<WaveSource>());
    for ( int i = 0; i < prof.uniform.size(); i++ ) {
        prof.uniform[i] = ui->table->item(i, 0)->checkState() == Qt::Checked;
        prof.value1[i] = mins[i]->value();
        prof.value2[i] = maxes[i]->value();
    }
    prof.samplingInterval = ui->interval->value();

    emit generate(prof);
}

void SamplingProfileDialog::on_btnAbort_clicked()
{
    session.samplingProfiler().abort();
}
