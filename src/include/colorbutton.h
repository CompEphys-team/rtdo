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


#ifndef COLORBUTTON_H
#define COLORBUTTON_H

#include <QToolButton>
#include <QColorDialog>

class ColorButton : public QToolButton
{
    Q_OBJECT
public:
    template <typename... Args> ColorButton(Args... args) :
        QToolButton(args...),
        color("black")
    {
        setColor(color);
        connect(this, &ColorButton::clicked, [=](){
            QColor ret = QColorDialog::getColor(color, parentWidget(), "Choose a colour");
            if ( ret.isValid() )
                setColor(ret);
        });
    }

    QColor color;

public slots:
    inline void setColor(QColor const& c)
    {
        QPalette p;
        p.setColor(QPalette::Button, c);
        p.setColor(QPalette::Window, c);
        setPalette(p);

        color = c;
        resizeEvent(nullptr);
        emit colorChanged(c);
    }

protected:
    inline void resizeEvent(QResizeEvent *ev)
    {
        QPixmap px(width(), height());
        px.fill(color);
        setIcon(px);
        setIconSize(size());
        if ( ev )
            QToolButton::resizeEvent(ev);
    }

signals:
    void colorChanged(QColor to);
};

#endif // COLORBUTTON_H
