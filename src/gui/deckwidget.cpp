#include "deckwidget.h"
#include "ui_deckwidget.h"

DeckWidget::DeckWidget(Session &session, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::DeckWidget),
    session(session)
{
    ui->setupUi(this);

    connect(ui->create, &QPushButton::clicked, this, &DeckWidget::create);
    connect(ui->decks, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, &DeckWidget::showExisting);
    connect(&session.wavesets(), &WavesetCreator::addedDeck, this, &DeckWidget::updateDecks);
    connect(&session.wavesets(), &WavesetCreator::addedSet, this, &DeckWidget::updateSets);

    QStringList labels;
    const std::vector<AdjustableParam> &params = session.project.model().adjustableParams;
    ui->sources->setRowCount(params.size());
    indices.resize(params.size());
    sources.resize(params.size());
    for ( size_t i = 0; i < params.size(); i++ ) {
        labels << QString::fromStdString(params[i].name);

        QSpinBox *index = new QSpinBox();
        ui->sources->setCellWidget(i, 0, index);
        indices[i] = index;

        QComboBox *source = new QComboBox();
        ui->sources->setCellWidget(i, 1, source);
        sources[i] = source;

        connect(source, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int i){
            if ( i >= 0 )
                index->setRange(0, source->currentData().value<WaveSource>().stimulations().size()-1);
        });
    }
    ui->sources->setVerticalHeaderLabels(labels);

    updateDecks();
}

DeckWidget::~DeckWidget()
{
    delete ui;
}

void DeckWidget::select(int paramIdx, WaveSource src, int stimIdx)
{
    int srcIdx = 0;
    std::vector<WaveSource> sets = session.wavesets().sources();
    for ( size_t i = 0; i < sets.size(); i++ ) {
        if ( sets[i] == src ) {
            srcIdx = i;
            break;
        }
    }

    sources[paramIdx]->setCurrentIndex(srcIdx);
    indices[paramIdx]->setValue(stimIdx);
}

void DeckWidget::clear()
{
    for ( size_t i = 0; i < sources.size(); i++ )
        sources[i]->setCurrentIndex(-1);
    for ( size_t i = 0; i < indices.size(); i++ )
        indices[i]->clear();
    ui->decks->setCurrentIndex(-1);
}

void DeckWidget::create()
{
    std::vector<WaveSource> src(sources.size());
    for ( size_t i = 0; i < sources.size(); i++ ) {
        if ( sources[i]->currentIndex() < 0 )
            return;
    }
    for ( size_t i = 0; i < sources.size(); i++ ) {
        WaveSource candidate = sources[i]->currentData().value<WaveSource>();
        candidate.waveno = indices[i]->value();
        src[i] = std::move(candidate);
    }
    WavesetCreator &creator = session.wavesets();
    session.queue(creator.actorName(), creator.actionDeck, "", new WaveDeck(session, src));
}

void DeckWidget::showExisting()
{
    int currentIndex = ui->decks->currentIndex();
    if ( currentIndex < 0 )
        return;
    const WaveDeck &deck = session.wavesets().decks().at(currentIndex);
    for ( size_t i = 0; i < sources.size(); i++ ) {
        sources[i]->setCurrentIndex(deck.sources().at(i).index());
        indices[i]->setValue(deck.sources().at(i).waveno);
    }
}

void DeckWidget::updateDecks()
{
    updateSets();
    for ( size_t i = ui->decks->count(); i < session.wavesets().decks().size(); i++ ) {
        ui->decks->addItem(WaveSource(session, WaveSource::Deck, i).prettyName());
    }
    ui->decks->setCurrentIndex(ui->decks->count()-1);
}

void DeckWidget::updateSets()
{
    std::vector<WaveSource> sets = session.wavesets().sources();
    std::vector<QString> labels(sets.size());
    std::vector<QVariant> variants(sets.size());
    for ( size_t i = 0; i < sets.size(); i++ ) {
        labels[i] = sets[i].prettyName();
        variants[i] = QVariant::fromValue(sets[i]);
    }

    for ( size_t i = 0; i < sources.size(); i++ ) {
        int currentIndex = sources[i]->currentIndex();
        sources[i]->clear();
        for ( size_t j = 0; j < sets.size(); j++ ) {
            sources[i]->addItem(labels.at(j), variants.at(j));
        }
        sources[i]->setCurrentIndex(currentIndex);
    }
}
