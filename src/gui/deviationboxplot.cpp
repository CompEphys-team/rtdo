#include "deviationboxplot.h"
#include "ui_deviationboxplot.h"

DeviationBoxPlot::DeviationBoxPlot(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::DeviationBoxPlot)
{
    ui->setupUi(this);
    connect(ui->boxplot_epoch, SIGNAL(valueChanged(int)), this, SLOT(replot()));
}

DeviationBoxPlot::~DeviationBoxPlot()
{
    delete ui;
}

void DeviationBoxPlot::init(Session *session)
{
    this->session = session;
}

void DeviationBoxPlot::setData(std::vector<FitInspector::Group> data, bool summarising)
{
    this->data = data;
    this->summarising = summarising;
    replot();
}

void DeviationBoxPlot::replot()
{
    ui->boxplot->clearPlottables();

    if ( data.empty() || (data.size() == 1 && data[0].fits.size() < 2) ) {
        ui->boxplot->replot();
        return;
    }
    int nParams = session->project.model().adjustableParams.size();

    // Find any parameters that aren't fitted at all
    std::vector<bool> fitted(nParams, false);
    int nFitted = 0;
    for ( const FitInspector::Group &g : data )
        for ( const FitInspector::Fit &f : g.fits )
            for ( int i = 0; i < nParams; i++ )
                fitted[i] = session->gaFitterSettings(f.fit().resultIndex).constraints[i] < 2;
    for ( const bool &b : fitted )
        nFitted += b;

    // Set up ticks
    int stride = data.size() + 1;
    int tickOffset = data.size() / 2;
    double barOffset = data.size()%2 ? 0 : 0.5;
    ui->boxplot->xAxis->setSubTicks(false);
    ui->boxplot->xAxis->setTickLength(0, 4);
    ui->boxplot->xAxis->grid()->setVisible(false);
    QSharedPointer<QCPAxisTickerText> textTicker(new QCPAxisTickerText);
    for ( int i = 0, fi = 0; i < nParams; fi += fitted[i], i++ ) {
        const AdjustableParam &p = session->project.model().adjustableParams[i];
        if ( fitted[i] )
            textTicker->addTick(tickOffset + fi*stride, QString::fromStdString(p.name) + (p.multiplicative ? "¹" : "²"));
    }
    textTicker->addTick(tickOffset + nFitted*stride, "joint");
    ui->boxplot->xAxis->setTicker(textTicker);
    ui->boxplot->yAxis->setLabel("Deviation (¹ %, ² % range)");
    ui->boxplot->legend->setVisible(summarising);

    // Enter data group by group
    quint32 epoch = ui->boxplot_epoch->value();
    for ( size_t i = 0; i < data.size(); i++ ) {
        QCPStatisticalBox *box = new QCPStatisticalBox(ui->boxplot->xAxis, ui->boxplot->yAxis);
        box->setName(data[i].label);
        QPen whiskerPen(Qt::SolidLine);
        whiskerPen.setCapStyle(Qt::FlatCap);
        box->setWhiskerPen(whiskerPen);
        box->setPen(QPen(data[i].color));
        QColor brushCol = data[i].color;
        brushCol.setAlphaF(0.3);
        box->setBrush(QBrush(brushCol));
        box->setWidth(0.8);

        std::vector<std::vector<double>> outcomes(nFitted + 1);
        for ( const FitInspector::Fit &f : data[i].fits ) {
            const GAFitter::Output &fit = f.fit();
            const std::vector<scalar> *params;
            if ( fit.final && (epoch == 0 || epoch >= fit.epochs) )
                params =& fit.finalParams;
            else if ( epoch >= fit.epochs )
                params =& fit.params[fit.epochs];
            else
                params =& fit.params[epoch];
            double value, total = 0;
            for ( int j = 0, fj = 0; j < nParams; fj += fitted[j], j++ ) {
                if ( !fitted[j] )
                    continue;
                const AdjustableParam &p = session->project.model().adjustableParams.at(j);
                if ( p.multiplicative ) {
                    value = 100 * std::fabs(1 - params->at(j) / fit.targets[j]);
                } else {
                    double range = session->gaFitterSettings(fit.resultIndex).constraints[j] == 1
                            ? (session->gaFitterSettings(fit.resultIndex).max[j] - session->gaFitterSettings(fit.resultIndex).min[j])
                            : (p.max - p.min);
                    value = 100 * std::fabs((params->at(j) - fit.targets[j]) / range);
                }
                total += value;
                outcomes[fj].push_back(value);
            }
            outcomes.back().push_back(total/nFitted);
        }
        for ( size_t j = 0; j < outcomes.size(); j++ ) {
            std::vector<double> q = Quantile(outcomes[j], {0, 0.25, 0.5, 0.75, 1});
            box->addData(j*stride + i + barOffset, q[0], q[1], q[2], q[3], q[4]);
        }
    }

    ui->boxplot->rescaleAxes();
    ui->boxplot->xAxis->setRange(-1, (nFitted+1)*stride - 1);

    ui->boxplot->replot();
}

void DeviationBoxPlot::on_boxplot_pdf_clicked()
{
    QString file = QFileDialog::getSaveFileName(this, "Select output file");
    if ( file.isEmpty() )
        return;
    if ( !file.endsWith(".pdf") )
        file.append(".pdf");
    ui->boxplot->savePdf(file, 0,0, QCP::epNoCosmetic, windowTitle(), file);
}
