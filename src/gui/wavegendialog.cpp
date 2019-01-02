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

    connect(ui->btnBubble, &QPushButton::clicked, [&](){ session.wavegen().search(Wavegen::bubble_action); });
    connect(ui->btnCluster, &QPushButton::clicked, [&](){ session.wavegen().search(Wavegen::cluster_action); });
    connect(ui->btnAbort, &QPushButton::clicked, &session, &Session::abort);

    ui->progressPlotter->init(session);
}

WavegenDialog::~WavegenDialog()
{
    delete ui;
}
