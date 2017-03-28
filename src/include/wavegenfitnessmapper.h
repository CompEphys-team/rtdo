#ifndef WAVEGENFITNESSMAPPER_H
#define WAVEGENFITNESSMAPPER_H

#include <QWidget>
#include "session.h"

namespace Ui {
class WavegenFitnessMapper;
}
class QButtonGroup;
class QDoubleSpinBox;
class QCheckBox;
class QCPColorMap;

class WavegenFitnessMapper : public QWidget
{
    Q_OBJECT

public:
    explicit WavegenFitnessMapper(Session &session, QWidget *parent = 0);
    ~WavegenFitnessMapper();

private slots:
    void updateCombo();
    void updateDimensions();

    void replot();

    void on_btnAdd_clicked();

private:
    Ui::WavegenFitnessMapper *ui;
    Session &session;

    QButtonGroup *groupx, *groupy;
    std::vector<QDoubleSpinBox*> mins, maxes;
    std::vector<QCheckBox*> collapse;

    std::unique_ptr<WavegenSelection> selection;
    QCPColorMap *colorMap;

    bool savedSelection;

    bool select(bool flattenToPlot);

    void initPlot();
};

#endif // WAVEGENFITNESSMAPPER_H
