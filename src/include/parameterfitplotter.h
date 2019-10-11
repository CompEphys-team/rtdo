/** RTDO: Closed loop neuron model fitting
 *  Copyright (C) 2019 Felix Kern <kernfel+github@gmail.com>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
**/


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

public slots:
    void replot();

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
