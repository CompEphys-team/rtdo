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

    struct ScoreStruct {
        double minF = __DBL_MAX__, maxF = 0, weightF;
        double minG = __DBL_MAX__, maxG = 0, weightG;
        double minA = __DBL_MAX__, maxA = 0, weightA;
        double sminF = __DBL_MAX__, smaxF = 0, sweightF;
        double sminG = __DBL_MAX__, smaxG = 0, sweightG;
        double sminA = __DBL_MAX__, smaxA = 0, sweightA;
        double norm, snorm;
    };

public:
    explicit SamplingProfilePlotter(Session &s, QWidget *parent = 0);
    ~SamplingProfilePlotter();

protected:
    double value(int i, int dimension,
                 const SamplingProfiler::Profile &prof,
                 const std::vector<MAPElite> &elites,
                 const std::vector<MAPEDimension> &dim,
                 const ScoreStruct &sstr);
    ScoreStruct getScoreStruct(const SamplingProfiler::Profile &prof,
                               const std::vector<MAPElite> &elites,
                               bool scoreF, bool scoreG, bool scoreA);

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

    static constexpr int nFixedColumns = 7;

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
