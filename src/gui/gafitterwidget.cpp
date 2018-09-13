#include "gafitterwidget.h"
#include "ui_gafitterwidget.h"
#include "cannedchannelassociationdialog.h"
#include "supportcode.h"

GAFitterWidget::GAFitterWidget(Session &session, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::GAFitterWidget),
    session(session),
    nQueued(0)
{
    ui->setupUi(this);

    ui->params_plotter->init(&session, true);

    connect(&session.wavesets(), SIGNAL(addedSet()), this, SLOT(updateDecks()));
    connect(&session.gaFitter(), SIGNAL(done()), this, SLOT(done()));
    connect(&session.gaFitter(), SIGNAL(progress(quint32)), this, SLOT(progress(quint32)));

    connect(&session.gaFitter(), SIGNAL(starting()), ui->response_plotter, SLOT(clear()));
    session.gaFitter().qV = &ui->response_plotter->qV;
    session.gaFitter().qI = &ui->response_plotter->qI;
    session.gaFitter().qO = &ui->response_plotter->qO;

    connect(&session, &Session::DAQDataChanged, this, [=](){
        ui->records->setEnabled(this->session.qDaqData().simulate < 0);
    });
    ui->records->setEnabled(this->session.qDaqData().simulate < 0);

    connect(&session, &Session::GAFitterSettingsChanged, this, &GAFitterWidget::updateDecks);
    updateDecks();

    connect(ui->finish, SIGNAL(clicked(bool)), &session.gaFitter(), SLOT(finish()), Qt::DirectConnection);
}

GAFitterWidget::~GAFitterWidget()
{
    delete ui;
}

void GAFitterWidget::updateDecks()
{
    ui->VCCreate->setEnabled(!session.wavesets().sources().empty());
    ui->start->setEnabled(!session.wavesets().sources().empty());

    int idx = ui->decks->currentIndex();
    ui->decks->clear();
    QStandardItemModel *model = qobject_cast<QStandardItemModel*>(ui->decks->model());
    int i = 0;
    for ( WaveSource &src : session.wavesets().sources() ) {
        ui->decks->addItem(src.prettyName(), QVariant::fromValue(src));
        if ( !session.qGaFitterSettings().useClustering ) {
            if ( i == idx && src.type != WaveSource::Deck )
                idx = -1;
            model->item(i++)->setFlags(src.type == WaveSource::Deck
                                       ? Qt::ItemIsEnabled | Qt::ItemIsSelectable
                                       : Qt::NoItemFlags);
        }
    }
    ui->decks->setCurrentIndex(idx);
}

void GAFitterWidget::progress(quint32 idx)
{
    ui->label_epoch->setText(QString("Epoch %1/%2").arg(idx).arg(session.gaFitterSettings().maxEpochs));
}

void GAFitterWidget::done()
{
    nQueued--;
    ui->label_epoch->setText("");
    ui->label_queued->setText(QString("%1 queued").arg(nQueued));
    if ( nQueued > 0 )
        ui->params_plotter->clear();
}

void GAFitterWidget::on_start_clicked()
{
    int currentSource = ui->decks->currentIndex();
    if ( currentSource < 0 )
        return;
    if ( nQueued == 0 ) {
        ui->params_plotter->clear();
        ui->label_epoch->setText("Starting...");
    }
    nQueued += ui->repeats->value();
    ui->label_queued->setText(QString("%1 queued").arg(nQueued));
    WaveSource src = session.wavesets().sources().at(currentSource);

    if ( session.qDaqData().simulate == -1 && ui->VCReadCfg->isChecked() ) {
        QString record = ui->VCRecord->text();
        record.replace(".atf", ".cfg");
        if ( QFileInfo(record).exists() )
            session.loadConfig(record);
    }

    for ( int i = 0; i < ui->repeats->value(); i++ )
        session.gaFitter().run(src, record.trimmed(), ui->VCReadCfg->isChecked());
}

void GAFitterWidget::on_abort_clicked()
{
    nQueued = 1;
    session.abort();
}

void GAFitterWidget::on_VCBrowse_clicked()
{
    QString file = QFileDialog::getOpenFileName(this, "Select voltage clamp recording", ui->VCRecord->text(), "*.atf");
    if ( file.isEmpty() )
        return;
    ui->VCRecord->setText(file);
}

void GAFitterWidget::on_VCChannels_clicked()
{
    if ( !QFile(ui->VCRecord->text()).exists() || ui->decks->currentIndex() < 0 )
        return;
    CannedDAQ daq(session);
    WaveSource src = session.wavesets().sources().at(ui->decks->currentIndex());
    std::vector<Stimulation> stims = session.gaFitter().sanitiseDeck(src.stimulations());
    daq.setRecord(stims, ui->VCRecord->text(), false, true);
    CannedChannelAssociationDialog *dlg = new CannedChannelAssociationDialog(session, &daq, this);
    dlg->open();
}

void GAFitterWidget::on_VCCreate_clicked()
{
    QString file = QFileDialog::getSaveFileName(this, "Select output file");
    if ( file.isEmpty() )
        return;
    if ( !file.endsWith(".atf") )
        file.append(".atf");

    std::ofstream of(file.toStdString());
    if ( !of.good() )
        return;
    WaveSource src = session.wavesets().sources().at(ui->decks->currentIndex());
    std::vector<Stimulation> stims = session.gaFitter().sanitiseDeck(src.stimulations(), true);

    int nTotal, nSamples, nBuffer;
    double dt = session.qRunData().dt;
    if ( session.qDaqData().filter.active )
        dt /= session.qDaqData().filter.samplesPerDt;
    CannedDAQ(session).getSampleNumbers(stims, dt, &nTotal, &nBuffer, &nSamples);

    // For details on the ATF format used as a stimulation file, see http://mdc.custhelp.com/app/answers/detail/a_id/17029
    // Note, the time column's values are ignored for stimulation.
    of << "ATF\t1.0\r\n";
    of << "2\t" << (stims.size()+1) << "\r\n";
    of << "\"Comment=RTDO VC stimulations. Project " << QDir(session.project.dir()).dirName() << " (" << session.project.model().name()
       << "), Session " << session.name() << ", " << src.prettyName() << "\"\r\n";
    of << "\"Use=Use as stimulation file with sampling interval " << (1000*dt) << " us (" << (1/dt) << " kHz) and "
       << nTotal << " total samples (" << (nTotal*dt/1000) << " s including buffers).\"\r\n";
    of << "\"Time (s)\"";
        for ( size_t i = 0; i < stims.size(); i++ ) {
            if ( src.type == WaveSource::Deck )
                of << "\t\"" << session.project.model().adjustableParams[i].name << " (mV)\"";
            else
                of << "\t\"stim " << i << " (mV)\"";
        }
        of << "\t\"noise_sample\"";
        of << "\r\n";

    for ( int i = 0; i < nBuffer; i++ ) {
        of << (i - nBuffer) * dt * 1e-3;
        for ( const Stimulation &stim : stims )
            of << '\t' << stim.baseV;
        of << "\r\n";
    }

    for ( int i = 0; i < nSamples; i++ ) {
        double t = i * dt;
        of << t * 1e-3;
        for ( const Stimulation &stim : stims )
            of << '\t' << getCommandVoltage(stim, t);
        of << "\r\n";
    }

    for ( int i = 0; i < nBuffer; i++ ) {
        of << (nSamples + i) * dt * 1e-3;
        for ( const Stimulation &stim : stims )
            of << '\t' << stim.baseV;
        of << "\r\n";
    }
}
