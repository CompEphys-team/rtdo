#ifndef GAFITTERWIDGET_H
#define GAFITTERWIDGET_H

#include <QWidget>
#include "session.h"

namespace Ui {
class GAFitterWidget;
}

class GAFitterWidget : public QWidget
{
    Q_OBJECT

public:
    explicit GAFitterWidget(Session &session, QWidget *parent = 0);
    ~GAFitterWidget();

signals:
    void startFitting();

private slots:
    void updateDecks();
    void progress(quint32 idx);
    void done();

    void on_start_clicked();

    void on_abort_clicked();

private:
    Ui::GAFitterWidget *ui;
    Session &session;
};

#endif // GAFITTERWIDGET_H
