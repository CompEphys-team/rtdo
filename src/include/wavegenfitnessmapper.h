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
class QComboBox;
class QSpinBox;
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

    void on_readMinFitness_clicked();

    void on_pdf_clicked();

private:
    Ui::WavegenFitnessMapper *ui;
    Session &session;

    QButtonGroup *groupx, *groupy;
    std::vector<QDoubleSpinBox*> mins, maxes;
    std::vector<QCheckBox*> collapse;
    std::vector<QComboBox*> pareto;
    std::vector<QSpinBox*> tolerance;

    std::unique_ptr<WavegenSelection> selection;
    QCPColorMap *colorMap;

    bool select(bool flattenToPlot);

    void initPlot();
};

#endif // WAVEGENFITNESSMAPPER_H
