#include "wavegendialog.h"
#include "ui_wavegendialog.h"
#include "config.h"

WavegenDialog::WavegenDialog(MetaModel &model, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::WavegenDialog),
    model(model),
    lib(model, Config::WavegenLibrary, Config::Run),
    wg(new Wavegen(lib, Config::Stimulation, Config::Wavegen))
{
    ui->setupUi(this);

    ui->btnPermute->setEnabled(lib.compileD.permute);
    for ( const AdjustableParam &p : model.adjustableParams )
        ui->cbSearch->addItem(QString::fromStdString(p.name));

    connect(ui->btnPermute, &QPushButton::clicked, [&](bool){
        wg->permute();
        ui->log->addItem("Parameters permuted.");
        ui->log->scrollToBottom();
    });
    connect(ui->btnSigadjust, &QPushButton::clicked, [&](bool){
        ui->log->addItem("Adjusting sigmas...");
        ui->log->scrollToBottom();
        setEnabled(false);
        repaint();
        wg->adjustSigmas();
        setEnabled(true);
        ui->log->addItem("Sigma adjustment complete.");
        ui->log->scrollToBottom();
    });
    connect(ui->btnSearchAll, &QPushButton::clicked, [&](bool){
        setEnabled(false);
        repaint();
        for ( size_t i = 0; i < model.adjustableParams.size(); i++ ) {
            ui->log->addItem(QString("Searching for %1...").arg(QString::fromStdString(model.adjustableParams[i].name)));
            ui->log->scrollToBottom();
            wg->search(i);
        }
        setEnabled(true);
        ui->log->addItem("Search complete.");
        ui->log->scrollToBottom();
    });
    connect(ui->btnSearchOne, &QPushButton::clicked, [&](bool){
        int i = ui->cbSearch->currentIndex();
        ui->log->addItem(QString("Searching for %1...").arg(QString::fromStdString(model.adjustableParams[i].name)));
        ui->log->scrollToBottom();
        setEnabled(false);
        repaint();
        wg->search(i);
        setEnabled(true);
        ui->log->addItem("Search complete.");
        ui->log->scrollToBottom();
    });
}

WavegenDialog::~WavegenDialog()
{
    delete ui;
    delete wg;
}
