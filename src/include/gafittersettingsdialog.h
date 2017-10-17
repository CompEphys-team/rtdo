#ifndef GAFITTERSETTINGSDIALOG_H
#define GAFITTERSETTINGSDIALOG_H

#include <QDialog>
#include "session.h"

namespace Ui {
class GAFitterSettingsDialog;
}
class QAbstractButton;

class GAFitterSettingsDialog : public QDialog
{
    Q_OBJECT

public:
    explicit GAFitterSettingsDialog(Session &s, int historicIndex = -1, QWidget *parent = 0);
    ~GAFitterSettingsDialog();

    void importData();
    void exportData();

private slots:
    void on_buttonBox_clicked(QAbstractButton *button);

private:
    Ui::GAFitterSettingsDialog *ui;
    Session &session;
    int historicIndex;

signals:
    void apply(GAFitterSettings);
};

#endif // GAFITTERSETTINGSDIALOG_H
