#ifndef PARAMETERFITPLOTTER_H
#define PARAMETERFITPLOTTER_H

#include <QWidget>
#include "session.h"
#include "qcustomplot.h"
#include "colorbutton.h"

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

protected:
    std::vector<int> getSelectedRows(QTableWidget *table);
    ColorButton *getGraphColorBtn(int row);
    ColorButton *getErrorColorBtn(int row);
    ColorButton *getGroupColorBtn(int row);

    void getSummary(std::vector<int> fits, //!< Integers identifying the fits to be used
                    std::function<double (const GAFitter::Output &, int)> value, //!< Function to extract the value to be analysed (eg param residual or error)
                    QVector<double> &mean, //!< Return: the mean value per epoch. Must be of appropriate size.
                    QVector<double> &meanPlusSEM, //!< Return: The SEM added onto the mean. Must be of appropriate size.
                    QVector<double> &median, //!< Return: the median value per epoch. Must be of appropriate size.
                    QVector<double> &max); //!< Return: The maximum value per epoch. Must be of appropriate size.
private slots:
    void setColumnCount(int n);
    void resizeTableRows(int, int, int size);
    void updateFits();

    void replot();
    void progress(quint32);
    void addFinal(const GAFitter::Output &);
    void xRangeChanged(QCPRange range);
    void errorRangeChanged(QCPRange range);
    void percentileRangeChanged(QCPRange range);

    void plotSummary();
    void addGroup(std::vector<int> group = {}, QString label = "");
    void removeGroup();

    void on_saveGroups_clicked();

    void on_loadGroups_clicked();

private:
    Ui::ParameterFitPlotter *ui;
    Session *session;

    std::vector<QCustomPlot*> plots;

    bool resizing, enslaved, summarising;

    std::vector<QColor> clipboard;

    std::vector<std::vector<int>> groups;
};

#endif // PARAMETERFITPLOTTER_H
