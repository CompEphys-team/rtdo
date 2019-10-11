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


#ifndef PROJECTSETTINGSDIALOG_H
#define PROJECTSETTINGSDIALOG_H

#include <QWidget>
#include "project.h"

namespace Ui {
class ProjectSettingsDialog;
}
class QPushButton;

class ProjectSettingsDialog : public QWidget
{
    Q_OBJECT

public:
    explicit ProjectSettingsDialog(Project *p, QWidget *parent = 0);
    ~ProjectSettingsDialog();

    void setProject(Project *project);

signals:
    void compile();

private slots:
    void on_buttonBox_accepted();

    void on_browseModel_clicked();

    void on_browseLocation_clicked();

    void keyPressEvent( QKeyEvent *e );

    void on_copy_clicked();

    void on_extraAdd_clicked();

    void on_extraRemove_clicked();

private:
    Ui::ProjectSettingsDialog *ui;
    Project *p;
    QPushButton *compileBtn;
};

#endif // PROJECTSETTINGSDIALOG_H
