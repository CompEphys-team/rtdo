#ifndef PCAPLOT_H
#define PCAPLOT_H

#include <QWidget>
#include "qcustomplot.h"
#include "universallibrary.h"

namespace Ui {
class PCAPlot;
}

class PCAPlot : public QWidget
{
    Q_OBJECT

public:
    explicit PCAPlot(QWidget *parent = 0);
    PCAPlot(Session &s, QWidget *parent = 0);
    ~PCAPlot();

    void init(const UniversalLibrary *lib);

public slots:
    void replot();

private slots:
    void compute();

    void on_pdf_clicked();

private:
    Ui::PCAPlot *ui;
    Session *session;
    const UniversalLibrary *lib;
    QSharedPointer<QCPGraphDataContainer> data;
    size_t x = 0, y = 1;
};

#endif // PCAPLOT_H
