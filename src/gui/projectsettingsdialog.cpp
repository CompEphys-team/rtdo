#include "projectsettingsdialog.h"
#include "ui_projectsettingsdialog.h"
#include <QFileDialog>
#include <QPushButton>
#include <QKeyEvent>

ProjectSettingsDialog::ProjectSettingsDialog(Project *p, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ProjectSettingsDialog)
{
    ui->setupUi(this);
    compileBtn = new QPushButton("Compile");
    ui->buttonBox->addButton(compileBtn, QDialogButtonBox::AcceptRole);
    setProject(p);
}

ProjectSettingsDialog::~ProjectSettingsDialog()
{
    delete ui;
}

void ProjectSettingsDialog::setProject(Project *project)
{
    p = project;
    ui->widget->setEnabled(!p->isFrozen());
    compileBtn->setEnabled(!p->isFrozen());
    switch ( p->method() ) {
    case IntegrationMethod::ForwardEuler: ui->integrationMethod->setCurrentIndex(0); break;
    case IntegrationMethod::RungeKutta4: ui->integrationMethod->setCurrentIndex(1);  break;
    }
    ui->dt->setValue(p->dt());
    ui->wgNumGroups->setValue(p->wgNumGroups());
    ui->profNumPairs->setValue(p->profNumPairs());
    ui->expNumCandidates->setValue(p->expNumCandidates());
    ui->modelFile->setText(p->modelfile());
    ui->projectLocation->setText(p->dir());
}

void ProjectSettingsDialog::on_buttonBox_accepted()
{
    switch ( ui->integrationMethod->currentIndex() ) {
    case 0: p->setMethod(IntegrationMethod::ForwardEuler); break;
    case 1: p->setMethod(IntegrationMethod::RungeKutta4);  break;
    }
    p->setDt(ui->dt->value());
    p->setWgNumGroups(ui->wgNumGroups->value());
    p->setProfNumPairs(ui->profNumPairs->value());
    p->setExpNumCandidates(ui->expNumCandidates->value());
    p->setModel(ui->modelFile->text());
    p->setLocation(ui->projectLocation->text() + "/project.dop");

    ui->widget->setEnabled(false);
    compileBtn->setEnabled(false);
    QApplication::setOverrideCursor(Qt::WaitCursor);
    QApplication::processEvents();
    p->compile();
    setProject(p);
    QApplication::restoreOverrideCursor();
    close();
}

void ProjectSettingsDialog::on_browseModel_clicked()
{
    QString file = QFileDialog::getOpenFileName(this, "Select model file", "", "Model files (*.xml)");
    if ( !file.isEmpty() )
        ui->modelFile->setText(file);
}

void ProjectSettingsDialog::on_browseLocation_clicked()
{
    QString loc = QFileDialog::getExistingDirectory(this, "Select project location");
    if ( !loc.isEmpty() )
        ui->projectLocation->setText(loc);
}

void ProjectSettingsDialog::keyPressEvent(QKeyEvent *e)
{
    if ( e->key() == Qt::Key_Escape )
        close();
    QWidget::keyPressEvent(e);
}
