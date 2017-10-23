#include "gafitterwidget.h"
#include "ui_gafitterwidget.h"

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
