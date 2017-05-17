#ifndef PROFILEPLOTTER_H
#define PROFILEPLOTTER_H

#include <QWidget>
#include "session.h"
#include "colorbutton.h"
#include <QCheckBox>

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
    void updateWaves();
    void replot();
    void rescale();

    void clearProfiles();
    void drawProfiles();
    void drawStats();

    void selectSubset();

private:
    Ui::ProfilePlotter *ui;
    Session &session;

    std::vector<QCheckBox*> includes;
    std::vector<ColorButton*> colors;

    bool tickingBoxes;

    void includeWave(size_t waveNo, bool on);
    void paintWave(size_t waveNo, QColor color);

    static constexpr int ValueColumn = 2;
};

#endif // PROFILEPLOTTER_H
