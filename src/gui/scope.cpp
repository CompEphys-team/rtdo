/** RTDO: Closed loop neuron model fitting
 *  Copyright (C) 2019 Felix Kern <kernfel+github@gmail.com>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
**/


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
    connect(ui->stop, &QPushButton::clicked, this, &Scope::stop);
    ui->plot->VC =& session.qRunData().VC;
}

Scope::~Scope()
{
    delete ui;
    delete daq;
}

void Scope::closeEvent(QCloseEvent *event)
{
    stop();
    ui->plot->clear();
    event->accept();
}

void Scope::on_start_clicked()
{
    if ( daq )
        stop();

    ui->start->setEnabled(false);
    ui->stop->setEnabled(true);

    daq = new RTMaybe::ComediDAQ(session, session.qSettings());
    daq->scope(2048);
    ui->plot->setDAQ(daq);
    ui->plot->start();
}

void Scope::stop()
{
    ui->start->setEnabled(true);
    ui->stop->setEnabled(false);

    ui->plot->stop();
    ui->plot->setDAQ(nullptr);
    delete daq;
    daq = nullptr;
}
