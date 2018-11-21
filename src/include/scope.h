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
