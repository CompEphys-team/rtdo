#include "wavegendialog.h"
#include "ui_wavegendialog.h"
#include "project.h"

WavegenDialog::WavegenDialog(Session *s, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::WavegenDialog),
    session(s),
    wavegen(s->wavegen()),
    abort(false)
{
    ui->setupUi(this);
    this->setWindowFlags(Qt::Window);
    initWG();
}

void WavegenDialog::initWG()
{
    ui->btnPermute->setEnabled(session->project.wgPermute());
    for ( const AdjustableParam &p : wavegen.lib.model.adjustableParams )
        ui->cbSearch->addItem(QString::fromStdString(p.name));

    connect(this, SIGNAL(permute()), &wavegen, SLOT(permute()));
    connect(this, SIGNAL(adjustSigmas()), &wavegen, SLOT(adjustSigmas()));
    connect(this, SIGNAL(search(int)), &wavegen, SLOT(search(int)));
    connect(&wavegen, SIGNAL(startedSearch(int)), this, SLOT(startedSearch(int)));
    connect(&wavegen, SIGNAL(searchTick(int)), this, SLOT(searchTick(int)));
    connect(&wavegen, SIGNAL(done(int)), this, SLOT(end(int)));

    connect(ui->btnPermute, &QPushButton::clicked, [&](bool){
        ui->log->addItem("Parameter permutation begins...");
        ui->log->scrollToBottom();
        actions.push_back("Parameter permutation");
        emit permute();
    });
    connect(ui->btnSigadjust, &QPushButton::clicked, [&](bool){
        ui->log->addItem("Sigma adjustment begins...");
        ui->log->scrollToBottom();
        actions.push_back("Sigma adjustment");
        emit adjustSigmas();
    });
    connect(ui->btnSearchAll, &QPushButton::clicked, [&](bool){
        for ( size_t i = 0; i < wavegen.lib.model.adjustableParams.size(); i++ ) {
            actions.push_back(QString("Search for %1").arg(QString::fromStdString(wavegen.lib.model.adjustableParams[i].name)));
            emit search(i);
        }
    });
    connect(ui->btnSearchOne, &QPushButton::clicked, [&](bool){
        int i = ui->cbSearch->currentIndex();
        actions.push_back(QString("Search for %1").arg(QString::fromStdString(wavegen.lib.model.adjustableParams[i].name)));
        emit search(i);
    });

    connect(ui->btnAbort, &QPushButton::clicked, [&](bool){
        wavegen.abort();
        abort = true;
    });
}

WavegenDialog::~WavegenDialog()
{
    delete ui;
}

void WavegenDialog::startedSearch(int)
{
    ui->log->addItem(QString("%1 begins...").arg(actions.front()));
    ui->log->addItem("");
    ui->log->scrollToBottom();
}

void WavegenDialog::searchTick(int i)
{
    ui->log->item(ui->log->count()-1)->setText(QString("%1 iterations...").arg(i));
}

void WavegenDialog::end(int)
{
    QString outcome = abort ? "aborted" : "complete";
    ui->log->addItem(QString("%1 %2.").arg(actions.front(), outcome));
    ui->log->scrollToBottom();
    if ( abort )
        actions.clear();
    else
        actions.pop_front();
    abort = false;
}
