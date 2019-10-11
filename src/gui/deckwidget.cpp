/** RTDO: Closed loop neuron model fitting
 *  Copyright (C) 2019 Felix Kern <kernfel+github@gmail.com>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
**/


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
        index->setRange(0, 1e9);

        QComboBox *source = new QComboBox();
        ui->sources->setCellWidget(i, 1, source);
        sources[i] = source;
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
    std::vector<WaveSource> sets = session.wavesets().sources();
    for ( size_t i = 0; i < sources.size(); i++ ) {
        if ( sources[i]->currentIndex() < 0 )
            return;
        if ( indices[i]->value() >= (int)sets[sources[i]->currentIndex()].stimulations().size() )
            return;
    }
    for ( size_t i = 0; i < sources.size(); i++ ) {
        src[i] = sets[sources[i]->currentIndex()];
        src[i].waveno = indices[i]->value();
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
    QStringList labels;
    for ( WaveSource &src : sets )
        labels << src.prettyName();
    for ( QComboBox *cb : sources ) {
        int currentIndex = cb->currentIndex();
        cb->clear();
        cb->insertItems(0, labels);
        cb->setCurrentIndex(currentIndex);
    }
}
