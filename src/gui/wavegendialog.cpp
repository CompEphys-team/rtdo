#include "wavegendialog.h"
#include "ui_wavegendialog.h"
#include "config.h"

WavegenDialog::WavegenDialog(MetaModel &model, QThread *thread, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::WavegenDialog),
    thread(thread),
    model(model),
    lib(model, Config::WavegenLibrary, Config::Run),
    wg(new Wavegen(lib, Config::Stimulation, Config::Wavegen)),
    abort(false)
{
    ui->setupUi(this);

    ui->btnPermute->setEnabled(lib.compileD.permute);
    for ( const AdjustableParam &p : model.adjustableParams )
        ui->cbSearch->addItem(QString::fromStdString(p.name));

    connect(this, SIGNAL(permute()), wg, SLOT(permute()));
    connect(this, SIGNAL(adjustSigmas()), wg, SLOT(adjustSigmas()));
    connect(this, SIGNAL(search(int)), wg, SLOT(search(int)));
    connect(wg, SIGNAL(startedSearch(int)), this, SLOT(startedSearch(int)));
    connect(wg, SIGNAL(searchTick(int)), this, SLOT(searchTick(int)));
    connect(wg, SIGNAL(done(int)), this, SLOT(end(int)));

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
        for ( size_t i = 0; i < model.adjustableParams.size(); i++ ) {
            actions.push_back(QString("Search for %1").arg(QString::fromStdString(model.adjustableParams[i].name)));
            emit search(i);
        }
    });
    connect(ui->btnSearchOne, &QPushButton::clicked, [&](bool){
        int i = ui->cbSearch->currentIndex();
        actions.push_back(QString("Search for %1").arg(QString::fromStdString(model.adjustableParams[i].name)));
        emit search(i);
    });

    connect(ui->btnAbort, &QPushButton::clicked, [&](bool){
        wg->abort();
        abort = true;
    });

    wg->moveToThread(thread);
}

WavegenDialog::~WavegenDialog()
{
    delete ui;
    delete wg;
}

void WavegenDialog::startedSearch(int param)
{
    ui->log->addItem(QString("%1 begins...").arg(actions.front()));
    ui->log->addItem("");
    ui->log->scrollToBottom();
}

void WavegenDialog::searchTick(int i)
{
    ui->log->item(ui->log->count()-1)->setText(QString("%1 iterations...").arg(i));
}

void WavegenDialog::end(int arg)
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
