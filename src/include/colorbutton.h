#ifndef COLORBUTTON_H
#define COLORBUTTON_H

#include <QToolButton>
#include <QColorDialog>
//#include <QPixmap>

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

        QPixmap px(width(), height());
        px.fill(c);
        setIcon(px);

        color = c;
        emit colorChanged(c);
    }

signals:
    void colorChanged(QColor to);
};

#endif // COLORBUTTON_H
