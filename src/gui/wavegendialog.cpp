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
