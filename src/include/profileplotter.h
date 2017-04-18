#ifndef PROFILEPLOTTER_H
#define PROFILEPLOTTER_H

#include <QWidget>
#include "session.h"

namespace Ui {
class ProfilePlotter;
}

class ProfilePlotter : public QWidget
{
    Q_OBJECT

public:
    explicit ProfilePlotter(Session &session, QWidget *parent = 0);
    ~ProfilePlotter();

private slots:
    void updateCombo();
    void updateTargets();
    void replot();
    void rescale();

private:
    Ui::ProfilePlotter *ui;
    Session &session;

    bool updatingCombo;
};

#endif // PROFILEPLOTTER_H
