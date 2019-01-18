#include "scope.h"
#include "ui_scope.h"
#include "session.h"
#include <QCloseEvent>

Scope::Scope(Session &session, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::Scope),
    session(session)
{
    ui->setupUi(this);
    connect(ui->clear, &QPushButton::clicked, ui->plot, &ResponsePlotter::clear);
    ui->plot->VC =& session.qRunData().VC;
}

Scope::~Scope()
{
    delete ui;
    delete daq;
}

void Scope::closeEvent(QCloseEvent *event)
{
    on_stop_clicked();
    ui->plot->clear();
    event->accept();
}

void Scope::on_start_clicked()
{
    if ( daq )
        on_stop_clicked();

    ui->start->setEnabled(false);
    ui->stop->setEnabled(true);

    daq = new RTMaybe::ComediDAQ(session, session.qSettings());
    daq->scope(2048);
    ui->plot->setDAQ(daq);
    ui->plot->start();
}

void Scope::on_stop_clicked()
{
    ui->start->setEnabled(true);
    ui->stop->setEnabled(false);

    ui->plot->stop();
    ui->plot->setDAQ(nullptr);
    delete daq;
    daq = nullptr;
}
