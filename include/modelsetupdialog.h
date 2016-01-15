#ifndef MODELSETUPDIALOG_H
#define MODELSETUPDIALOG_H

#include <QDialog>

namespace Ui {
class ModelSetupDialog;
}

class ModelSetupDialog : public QDialog
{
    Q_OBJECT

public:
    explicit ModelSetupDialog(QWidget *parent = 0);
    ~ModelSetupDialog();

public slots:
    void open();
    void accept();

private slots:
    void on_modelfile_browse_clicked();

    void on_outdir_browse_clicked();

private:
    Ui::ModelSetupDialog *ui;
};

#endif // MODELSETUPDIALOG_H
