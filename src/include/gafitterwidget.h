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


#ifndef GAFITTERWIDGET_H
#define GAFITTERWIDGET_H

#include <QWidget>
#include "session.h"

namespace Ui {
class GAFitterWidget;
}

class GAFitterWidget : public QWidget
{
    Q_OBJECT

public:
    explicit GAFitterWidget(Session &session, QWidget *parent = 0);
    ~GAFitterWidget();
    void unqueue(int n);

private slots:
    void updateDecks();
    void progress(quint32 idx);
    void done();

    void on_start_clicked();

    void on_abort_clicked();

    void on_VCBrowse_clicked();

    void on_VCChannels_clicked();

    void on_VCCreate_clicked();

    void on_cl_run_clicked();

    void on_validate_clicked();

private:
    Ui::GAFitterWidget *ui;
    Session &session;
    int nQueued;
};

#endif // GAFITTERWIDGET_H
