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

    connect(&session.wavesets(), SIGNAL(addedDeck()), this, SLOT(updateDecks()));
    connect(this, SIGNAL(startFitting(WaveSource)), &session.gaFitter(), SLOT(run(WaveSource)));
    connect(&session.gaFitter(), SIGNAL(done()), this, SLOT(done()));
    connect(&session.gaFitter(), SIGNAL(progress(quint32)), this, SLOT(progress(quint32)));

    connect(&session.gaFitter(), SIGNAL(starting()), ui->response_plotter, SLOT(clear()));
    session.gaFitter().qV = &ui->response_plotter->qV;
    session.gaFitter().qI = &ui->response_plotter->qI;
    session.gaFitter().qO = &ui->response_plotter->qO;

    updateDecks();

    connect(ui->finish, SIGNAL(clicked(bool)), &session.gaFitter(), SLOT(finish()), Qt::DirectConnection);
}

GAFitterWidget::~GAFitterWidget()
{
    delete ui;
}

void GAFitterWidget::updateDecks()
{
    for ( size_t i = ui->decks->count(); i < session.wavesets().decks().size(); i++ )
        ui->decks->addItem(WaveSource(session, WaveSource::Deck, i).prettyName());
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
    int currentDeck = ui->decks->currentIndex();
    if ( currentDeck < 0 )
        return;
    if ( nQueued == 0 ) {
        ui->params_plotter->clear();
        ui->label_epoch->setText("Starting...");
    }
    nQueued += ui->repeats->value();
    ui->label_queued->setText(QString("%1 queued").arg(nQueued));
    WaveSource deck(session, WaveSource::Deck, currentDeck);
    for ( int i = 0; i < ui->repeats->value(); i++ )
        emit startFitting(deck);
}

void GAFitterWidget::on_abort_clicked()
{
    nQueued = 1;
    session.abort();
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
    WaveDeck deck = session.wavesets().decks().at(ui->decks->currentIndex());
    std::vector<Stimulation> stims = session.gaFitter().sanitiseDeck(deck.stimulations(), session.qRunData());

    int nTotal, nSamples, nBuffer;
    double dt = session.project.dt();
    if ( session.qDaqData().filter.active )
        dt /= session.qDaqData().filter.samplesPerDt;
    CannedDAQ(session).getSampleNumbers(stims, dt, &nTotal, &nBuffer, &nSamples);

    // For details on the ATF format used as a stimulation file, see http://mdc.custhelp.com/app/answers/detail/a_id/17029
    // Note, the time column's values are ignored for stimulation.
    of << "ATF\t1.0\r\n";
    of << "2\t" << (stims.size()+1) << "\r\n";
    of << "\"Comment=RTDO VC stimulations. Project " << QDir(session.project.dir()).dirName() << " (" << session.project.model().name()
       << "), Session " << session.name() << ", " << WaveSource(session, WaveSource::Deck, ui->decks->currentIndex()).prettyName() << "\"\r\n";
    of << "\"Use=Use as stimulation file with sampling interval " << (1000*dt) << " us and " << nTotal << " total samples.\"\r\n";
    of << "\"Time (s)\"";
        for ( const AdjustableParam &p : session.project.model().adjustableParams )
            of << "\t\"" << p.name << " (mV)\"";
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
