#ifndef FIXPARAMDIALOG_H
#define FIXPARAMDIALOG_H

#include <QDialog>

namespace Ui {
class FixParamDialog;
}

class FixParamDialog : public QDialog
{
    Q_OBJECT

public:
    explicit FixParamDialog(QWidget *parent = 0);
    ~FixParamDialog();

    int param;
    double value;

public slots:
    void accept();

private slots:
    void on_pushButton_clicked();

private:
    Ui::FixParamDialog *ui;
};

#endif // FIXPARAMDIALOG_H
