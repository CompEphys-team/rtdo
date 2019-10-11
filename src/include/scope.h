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


#ifndef SCOPE_H
#define SCOPE_H

#include <QWidget>
#include "comedidaq.h"

namespace Ui {
class Scope;
}

class Scope : public QWidget
{
    Q_OBJECT

public:
    explicit Scope(Session &session, QWidget *parent = 0);
    ~Scope();

protected:
    void closeEvent(QCloseEvent *event);

private slots:
    void on_start_clicked();
    void on_stop_clicked();

private:
    Ui::Scope *ui;
    Session &session;
    RTMaybe::ComediDAQ *daq = nullptr;
};

#endif // SCOPE_H
