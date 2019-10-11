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


#ifndef DECKWIDGET_H
#define DECKWIDGET_H

#include <QWidget>
#include <QComboBox>
#include <QSpinBox>
#include "session.h"

namespace Ui {
class DeckWidget;
}

class DeckWidget : public QWidget
{
    Q_OBJECT

public:
    explicit DeckWidget(Session &session, QWidget *parent = 0);
    ~DeckWidget();

public slots:
    void select(int paramIdx, WaveSource src, int stimIdx);
    void clear();

private slots:
    void create();
    void showExisting();
    void updateDecks();
    void updateSets();

private:
    Ui::DeckWidget *ui;
    Session &session;

    std::vector<QComboBox*> sources;
    std::vector<QSpinBox*> indices;
};

#endif // DECKWIDGET_H
