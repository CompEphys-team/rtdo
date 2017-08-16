#ifndef DAQDIALOG_H
#define DAQDIALOG_H

#include <QDialog>
#include "session.h"

namespace Ui {
class DAQDialog;
}
class QComboBox;
class QDoubleSpinBox;
class QAbstractButton;

class DAQDialog : public QDialog
{
    Q_OBJECT

    struct ChannelUI
    {
        QComboBox *channel;
        QComboBox *range;
        QComboBox *aref;
        QDoubleSpinBox *factor;
        QDoubleSpinBox *offset;
    };

public:
    explicit DAQDialog(Session &s, QWidget *parent = 0);
    ~DAQDialog();

public slots:
    void importData();
    DAQData exportData();

signals:
    void apply(DAQData);

protected slots:
    void updateChannelCapabilities(int tab = -1, bool checkDevice = true);
    void updateSingleChannelCapabilities(void *vdev, int subdev, ChannelUI &cui);

private slots:
    void on_buttonBox_clicked(QAbstractButton *button);

private:
    Ui::DAQDialog *ui;
    Session &session;

    ChannelUI chanUI[4];
};

#endif // DAQDIALOG_H
