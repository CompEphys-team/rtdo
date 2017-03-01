#include "profiledialog.h"
#include "ui_profiledialog.h"

ProfileDialog::ProfileDialog(ExperimentLibrary &lib, QThread *thread, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ProfileDialog),
    thread(thread),
    lib(lib),
    profiler(lib),
    selections(nullptr)
{
    ui->setupUi(this);
    setWindowFlags(Qt::Window);
}

ProfileDialog::~ProfileDialog()
{
    delete ui;
}

void ProfileDialog::selectionsChanged(WavegenDialog *dlg)
{
    selections =& dlg->selections;
    ui->cbSelection->clear();
    for ( WavegenDialog::Selection const& sel : *selections )
        ui->cbSelection->addItem(dlg->name(sel));
}
