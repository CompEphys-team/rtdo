#ifndef DEVIATIONBOXPLOT_H
#define DEVIATIONBOXPLOT_H

#include <QWidget>
#include "fitinspector.h"

namespace Ui {
class DeviationBoxPlot;
}

class DeviationBoxPlot : public QWidget
{
    Q_OBJECT

public:
    explicit DeviationBoxPlot(QWidget *parent = 0);
    ~DeviationBoxPlot();

    void init(Session *session);

    void setData(std::vector<FitInspector::Group> data, bool summarising);

public slots:
    void replot();

private slots:
    void on_boxplot_pdf_clicked();

private:
    Ui::DeviationBoxPlot *ui;
    Session *session;

    bool summarising = false;
    std::vector<FitInspector::Group> data;
};

#endif // DEVIATIONBOXPLOT_H
