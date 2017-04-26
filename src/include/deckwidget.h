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
