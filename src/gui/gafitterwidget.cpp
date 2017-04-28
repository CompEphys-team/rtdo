#include "gafitterwidget.h"
#include "ui_gafitterwidget.h"

GAFitterWidget::GAFitterWidget(Session &session, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::GAFitterWidget),
    session(session)
{
    ui->setupUi(this);

    connect(&session.wavesets(), SIGNAL(addedDeck()), this, SLOT(updateDecks()));
    connect(this, SIGNAL(startFitting()), &session.gaFitter(), SLOT(run()));
    connect(&session.gaFitter(), SIGNAL(done()), this, SLOT(done()));
    connect(&session.gaFitter(), SIGNAL(progress(quint32)), this, SLOT(progress(quint32)));

    ui->params_plotter->init(&session, true);

    updateDecks();
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
    ui->start->setText(QString("%1 iterations...").arg(idx));
    // ...
}

void GAFitterWidget::done()
{
    ui->start->setText("Start");
    ui->start->setEnabled(true);
}

void GAFitterWidget::on_start_clicked()
{
    int currentDeck = ui->decks->currentIndex();
    if ( currentDeck < 0 )
        return;
    ui->start->setEnabled(false);
    ui->params_plotter->clear();
    session.gaFitter().stageDeck(WaveSource(session, WaveSource::Deck, currentDeck));
    emit startFitting();
}

void GAFitterWidget::on_abort_clicked()
{
    session.gaFitter().abort();
    ui->start->setText("Start");
    ui->start->setEnabled(true);
}
