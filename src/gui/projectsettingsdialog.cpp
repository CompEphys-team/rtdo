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
    ui->expNumCandidates->setValue(p->expNumCandidates());
    ui->modelFile->setText(p->modelfile());
    ui->projectLocation->setText(p->dir());
    ui->extraModels->setRowCount(p->extraModelFiles().size());
    int i = 0;
    for ( const QString &m : p->extraModelFiles() )
        ui->extraModels->setItem(i++, 0, new QTableWidgetItem(m));
}

void ProjectSettingsDialog::on_buttonBox_accepted()
{
    p->setExpNumCandidates(ui->expNumCandidates->value());
    p->setModel(ui->modelFile->text());
    p->setLocation(ui->projectLocation->text() + "/project.dop");

    std::vector<QString> extraModels;
    extraModels.reserve(ui->extraModels->rowCount());
    for ( int i = 0; i < ui->extraModels->rowCount(); i++ ) {
        QString path = ui->extraModels->item(i, 0)->text();
        if ( !path.isEmpty() )
            extraModels.push_back(path);
    }
    p->setExtraModels(extraModels);

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

void ProjectSettingsDialog::on_copy_clicked()
{
    QString file = QFileDialog::getOpenFileName(this, "Select project file", "", "Project files (*.dop)");
    if ( !file.isEmpty() ) {
        p->loadSettings(file);
        setProject(p);
    }
}

void ProjectSettingsDialog::on_extraAdd_clicked()
{
    QString file = QFileDialog::getOpenFileName(this, "Select model file", "", "Model files (*.xml)");
    if ( !file.isEmpty() ) {
        int r = ui->extraModels->rowCount();
        ui->extraModels->insertRow(r);
        ui->extraModels->setItem(r, 0, new QTableWidgetItem(file));
    }
}

void ProjectSettingsDialog::on_extraRemove_clicked()
{
    auto selected = ui->extraModels->selectedItems();
    for ( int i = selected.size()-1; i >= 0; i-- )
        ui->extraModels->removeRow(selected[i]->row());
}
