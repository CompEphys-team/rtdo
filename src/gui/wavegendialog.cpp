#include "wavegendialog.h"
#include "ui_wavegendialog.h"
#include "project.h"

WavegenDialog::WavegenDialog(Session &s, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::WavegenDialog),
    session(s)
{
    ui->setupUi(this);
    this->setWindowFlags(Qt::Window);

    for ( const AdjustableParam &p : session.wavegen().lib.model.adjustableParams )
        ui->cbSearch->addItem(QString::fromStdString(p.name));

    connect(ui->btnSigadjust, &QPushButton::clicked, &session.wavegen(), &Wavegen::adjustSigmas);
    connect(ui->btnSearchAll, &QPushButton::clicked, [&](bool){
        for ( size_t i = 0; i < session.wavegen().lib.model.adjustableParams.size(); i++ ) {
            session.wavegen().search(i);
        }
    });
    connect(ui->btnSearchOne, &QPushButton::clicked, [&](bool){
        session.wavegen().search(ui->cbSearch->currentIndex());
    });

    connect(ui->btnEE, &QPushButton::clicked, &session.wavegen(), &Wavegen::clusterSearch);

    connect(ui->btnAbort, &QPushButton::clicked, &session, &Session::abort);

    ui->progressPlotter->init(session);
}

WavegenDialog::~WavegenDialog()
{
    delete ui;
}
