#ifndef WAVEGENDIALOG_H
#define WAVEGENDIALOG_H

#include <QDialog>
#include <QThread>
#include <QButtonGroup>
#include "qcustomplot.h"
#include "wavegen.h"

namespace Ui {
class WavegenDialog;
}

class WavegenDialog : public QDialog
{
    Q_OBJECT

public:
    explicit WavegenDialog(Project *p, QThread *thread, QWidget *parent = 0);
    ~WavegenDialog();

    struct Selection { //!< A selection is a 2D set of elites. Unselected dimensions are collapsed to the best-performing stimulation in range.
        int param;
        std::vector<MAPElite> elites; //!< All selected elites (access: [ix + nx*iy])
        std::vector<double> min, max; //!< Ranges for each MAPE dimension
        int cx, cy; //!< Dimension indices for the x and y axis
        int nx, ny; //!< Number of bins along each axis.
    };
    QString name(const Selection &sel) const;
    std::vector<Selection> selections;

signals:
    void permute();
    void adjustSigmas();
    void search(int param);

    void selectionsChanged();

private slots:
    void end(int);
    void startedSearch(int);
    void searchTick(int);

    void replot(bool doSelect = true);
    void setPlotMinMaxSteps(int);
    void on_btnAddToSel_clicked();
    void on_cbSelections_currentIndexChanged(int index);

private:
    Ui::WavegenDialog *ui;
    Project *project;
    QThread *thread;

    MetaModel &model;
    WavegenLibrary &lib;
    Wavegen *wg;

    std::list<QString> actions;

    bool abort;

    QButtonGroup *groupx, *groupy;
    std::vector<QDoubleSpinBox*> mins, maxes;

    QCPColorMap *colorMap;

    Selection currentSelection;

    void initWG();
    void initPlotControls();
    void refreshPlotControls();

    bool select();
};

#endif // WAVEGENDIALOG_H
