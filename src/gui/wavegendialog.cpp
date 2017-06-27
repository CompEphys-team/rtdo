#include "wavegendialog.h"
#include "ui_wavegendialog.h"
#include "project.h"

WavegenDialog::WavegenDialog(Session &s, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::WavegenDialog),
    session(s),
    abort(false)
{
    ui->setupUi(this);
    this->setWindowFlags(Qt::Window);

    for ( const AdjustableParam &p : session.wavegen().lib.model.adjustableParams )
        ui->cbSearch->addItem(QString::fromStdString(p.name));

    connect(this, SIGNAL(adjustSigmas()), &session.wavegen(), SLOT(adjustSigmas()));
    connect(this, SIGNAL(search(int)), &session.wavegen(), SLOT(search(int)));
    connect(&session.wavegen(), SIGNAL(startedSearch(int)), this, SLOT(startedSearch(int)));
    connect(&session.wavegen(), SIGNAL(searchTick(int)), this, SLOT(searchTick(int)));
    connect(&session.wavegen(), SIGNAL(done(int)), this, SLOT(end(int)));

    connect(ui->btnSigadjust, &QPushButton::clicked, [&](bool){
        ui->log->addItem("Sigma adjustment begins...");
        ui->log->scrollToBottom();
        actions.push_back("Sigma adjustment");
        emit adjustSigmas();
    });
    connect(ui->btnSearchAll, &QPushButton::clicked, [&](bool){
        for ( size_t i = 0; i < session.wavegen().lib.model.adjustableParams.size(); i++ ) {
            actions.push_back(QString("Search for %1").arg(QString::fromStdString(session.wavegen().lib.model.adjustableParams[i].name)));
            emit search(i);
        }
    });
    connect(ui->btnSearchOne, &QPushButton::clicked, [&](bool){
        int i = ui->cbSearch->currentIndex();
        actions.push_back(QString("Search for %1").arg(QString::fromStdString(session.wavegen().lib.model.adjustableParams[i].name)));
        emit search(i);
    });

    connect(ui->btnAbort, &QPushButton::clicked, [&](bool){
        session.wavegen().abort();
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
