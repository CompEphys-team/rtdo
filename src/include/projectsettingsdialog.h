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

private:
    Ui::ProjectSettingsDialog *ui;
    Project *p;
    QPushButton *compileBtn;
};

#endif // PROJECTSETTINGSDIALOG_H
