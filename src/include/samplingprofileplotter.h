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
    std::function<double(int)> valueFunction(int dimension,
                                             const SamplingProfiler::Profile &prof,
                                             const std::vector<MAPElite> &elites,
                                             const std::vector<MAPEDimension> &dim);

protected slots:
    void updateProfiles();
    void setProfile(int);
    void replot(bool discardSelection = false);
    void hideUnselected();
    void showAll();

private:
    Ui::SamplingProfilePlotter *ui;
    Session &session;

    static constexpr int nFixedColumns = 4;

    bool updating;

    struct DataPoint {
        double key, value;
        size_t idx;
        bool selected;
    };
    std::vector<DataPoint> plottedPoints;
};

#endif // SAMPLINGPROFILEPLOTTER_H
