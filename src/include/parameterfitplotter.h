#ifndef PARAMETERFITPLOTTER_H
#define PARAMETERFITPLOTTER_H

#include <QWidget>
#include "session.h"
#include "qcustomplot.h"
#include "fitinspector.h"
#include "filter.h"

namespace Ui {
class ParameterFitPlotter;
}

class ParameterFitPlotter : public QWidget
{
    Q_OBJECT

public:
    explicit ParameterFitPlotter(QWidget *parent = 0);
    ParameterFitPlotter(Session &session, QWidget *parent = 0);
    ~ParameterFitPlotter();

    void init(Session *session, bool enslave);
    void clear();

    void setData(std::vector<FitInspector::Group> data, bool summarising);

protected:

    void getSummary(std::vector<FitInspector::Fit> fits, //!< The fits to be used
                    std::function<double (const GAFitter::Output &, int)> value, //!< Function to extract the value to be analysed (eg param residual or error)
                    QVector<double> &mean, //!< Return: the mean value per epoch. Must be of appropriate size.
                    QVector<double> &meanPlusSEM, //!< Return: The SEM added onto the mean. Must be of appropriate size.
                    QVector<double> &median, //!< Return: the median value per epoch. Must be of appropriate size.
                    QVector<double> &max, //!< Return: The maximum value per epoch. Must be of appropriate size.
                    Filter *filter = nullptr);
    QCPGraph *addGraph(QCPAxis *x, QCPAxis *y, const QColor &col,
                       const QVector<double> &keys, const QVector<double> &values,
                       const QString &layer, bool visible);

protected slots:
    void resizeEvent(QResizeEvent *event);
    void resizePanel();
    void setGridAndAxVisibility();
    void clearPlotLayout();
    void buildPlotLayout();

private slots:

    void replot();
    void plotIndividual();
    void progress(quint32);
    void addFinal(const GAFitter::Output &);
    void xRangeChanged(QCPRange range);
    void errorRangeChanged(QCPRange range);
    void percentileRangeChanged(QCPRange range);

    void plotSummary();

    void on_pdf_clicked();

private:
    Ui::ParameterFitPlotter *ui;
    Session *session;

    std::vector<QCPAxisRect*> axRects;

    bool enslaved, summarising;
    std::vector<FitInspector::Group> data;
};

#endif // PARAMETERFITPLOTTER_H
