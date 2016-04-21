#include "fixparamdialog.h"
#include "ui_fixparamdialog.h"
#include "config.h"

FixParamDialog::FixParamDialog(QWidget *parent) :
    QDialog(parent),
    param(-1),
    value(0.0),
    ui(new Ui::FixParamDialog)
{
    ui->setupUi(this);
    for ( auto p : config->model.obj->adjustableParams() ) {
        ui->comboBox->addItem(QString::fromStdString(p.name));
    }
}

FixParamDialog::~FixParamDialog()
{
    delete ui;
}

void FixParamDialog::accept()
{
    param = ui->comboBox->currentIndex();
    value = ui->doubleSpinBox->value();
    QDialog::accept();
}

void FixParamDialog::on_pushButton_clicked()
{
    int p = ui->comboBox->currentIndex();
    double v = config->model.obj->adjustableParams().at(p).initial;
    ui->doubleSpinBox->setValue(v);
}
