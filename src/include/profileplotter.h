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
    void updateProfiles();
    void updateTargets();
    void replot();
    void rescale();

    void clearProfiles();
    void drawProfiles();
    void drawStats();

private:
    Ui::ProfilePlotter *ui;
    Session &session;
};

#endif // PROFILEPLOTTER_H
