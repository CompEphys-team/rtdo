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


#ifndef FITINSPECTOR_H
#define FITINSPECTOR_H

#include <QWidget>
#include <QTableWidget>
#include "colorbutton.h"
#include "session.h"

namespace Ui {
class FitInspector;
}

class FitInspector : public QWidget
{
    Q_OBJECT

public:
    explicit FitInspector(Session &session, QWidget *parent = 0);
    ~FitInspector();

    void replot(bool summarising);

    struct Fit
    {
        QString label;
        QColor color, errColor;
        int idx;
        const GAFitter *gaf;
        const inline GAFitter::Output &fit() const { return gaf->results().at(idx); }
    };

    struct Group
    {
        QString label;
        QColor color;
        std::vector<Fit> fits;
    };

protected:
    std::vector<int> getSelectedRows(QTableWidget *table);
    ColorButton *getGraphColorBtn(int row);
    ColorButton *getErrorColorBtn(int row);
    ColorButton *getGroupColorBtn(int row);

private slots:
    void addGroup(std::vector<int> group = {}, QString label = "");
    void removeGroup();
    void on_saveGroups_clicked();
    void on_loadGroups_clicked();

    void updateFits();

private:
    Ui::FitInspector *ui;
    Session &session;

    std::vector<QColor> clipboard;

    std::vector<std::vector<int>> groups;
};

#endif // FITINSPECTOR_H
