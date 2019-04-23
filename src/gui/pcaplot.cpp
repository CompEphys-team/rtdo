#include "pcaplot.h"
#include "ui_pcaplot.h"
#include "session.h"
#include "populationsaver.h"

PCAPlot::PCAPlot(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::PCAPlot),
    data(QSharedPointer<QCPGraphDataContainer>::create())
{
    ui->setupUi(this);
    ui->controls->hide();
}

PCAPlot::PCAPlot(Session &s, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::PCAPlot),
    session(&s),
    data(QSharedPointer<QCPGraphDataContainer>::create())
{
    ui->setupUi(this);
    for ( size_t i = 0; i < session->gaFitter().results().size(); i++ )
        ui->fits->addItem(QString("Fit %1 (%2)").arg(i).arg(session->gaFitter().results().at(i).resultIndex, 4, 10, QChar('0')));
    connect(&session->gaFitter(), &GAFitter::done, [=](){
        ui->fits->addItem(QString("Fit %1 (%2)").arg(ui->fits->count()).arg(session->gaFitter().results().back().resultIndex, 4, 10, QChar('0')));
    });
    connect(ui->fits, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int i){
        ui->epoch->setMaximum(session->gaFitter().results().at(i).epochs);
        compute();
    });
    connect(ui->epoch, static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), this, &PCAPlot::compute);

    init(&session->project.universal());
}

PCAPlot::~PCAPlot()
{
    delete ui;
}

void PCAPlot::init(const UniversalLibrary *lib)
{
    this->lib = lib;

    QCPGraph *g = ui->plot->addGraph();
    g->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, 2));
    g->setLineStyle(QCPGraph::lsNone);
    g->setData(data);
}

void PCAPlot::replot()
{
    if ( !lib )
        return;

    if ( data->size() != lib->PCA_TL_size ) {
        data->set(QVector<QCPGraphData>(lib->PCA_TL_size, {0,0}));
    }

    size_t i = 0;
    for ( QCPGraphData &p : *data ) {
        p.key = lib->PCA_TL[i];
        p.value = lib->PCA_TL[lib->PCA_TL_size + i];
        ++i;
    }
    data->sort();
    ui->plot->rescaleAxes();
    ui->plot->replot();
}

void PCAPlot::compute()
{
    if ( session->busy() || ui->fits->currentIndex() < 0 || !session )
        return;

    QFile basefile(session->gaFitter().getBaseFilePath(ui->fits->currentIndex()));
    PopLoader loader(basefile, session->project.universal());
    loader.load(ui->epoch->value(), session->project.universal());
    session->project.universal().pushParams();
    std::vector<scalar> singular_values = session->project.universal().get_params_principal_components(2);
    for ( const scalar &s : singular_values )
        std::cout << s << '\t';
    std::cout << std::endl;
    replot();
}
