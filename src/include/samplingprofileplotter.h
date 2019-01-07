#ifndef SAMPLINGPROFILEPLOTTER_H
#define SAMPLINGPROFILEPLOTTER_H

#include <QWidget>
#include "session.h"

namespace Ui {
class SamplingProfilePlotter;
}

class SamplingProfilePlotter : public QWidget
{
    Q_OBJECT

public:
    explicit SamplingProfilePlotter(Session &s, QWidget *parent = 0);
    ~SamplingProfilePlotter();

protected:
    double value(int i, int dimension,
                 const SamplingProfiler::Profile &prof,
                 const std::vector<MAPElite> &elites,
                 const std::vector<MAPEDimension> &dim);

protected slots:
    void updateTable();
    void updateProfiles();
    void setProfile(int);
    void replot(bool discardSelection = false, bool showAll = false);
    void hideUnselected();
    void showAll();

private slots:
    void on_pdf_clicked();

private:
    Ui::SamplingProfilePlotter *ui;
    Session &session;

    static constexpr int nFixedColumns = 8;

    bool updating;

    struct DataPoint {
        double key, value;
        size_t idx;
        bool selected;
        bool hidden;
    };
    std::vector<DataPoint> points;
};

#endif // SAMPLINGPROFILEPLOTTER_H
