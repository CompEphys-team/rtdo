#include "fitinspector.h"
#include "ui_fitinspector.h"

#include "rundatadialog.h"
#include "daqdialog.h"
#include "gafittersettingsdialog.h"

FitInspector::FitInspector(Session &session, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::FitInspector),
    session(session)
{
    ui->setupUi(this);

    connect(ui->sepcols, &QPushButton::clicked, [=](bool on) {
        for ( int i = 0; i < ui->fits->rowCount(); i++ ) {
            getGraphColorBtn(i)->setColor(on ? QColorDialog::standardColor(i%20) : Qt::blue);
            getErrorColorBtn(i)->setColor(on ? QColorDialog::standardColor(i%20 + 21) : Qt::magenta);
        }
        replot(false);
    });
    connect(ui->copy, &QPushButton::clicked, [=]() {
        std::vector<int> rows = getSelectedRows(ui->fits);
        clipboard.clear();
        clipboard.reserve(2*rows.size());
        for ( int row : rows ) {
            clipboard.push_back(getGraphColorBtn(row)->color);
            clipboard.push_back(getErrorColorBtn(row)->color);
        }
    });
    connect(ui->paste, &QPushButton::clicked, [=]() {
        std::vector<int> rows = getSelectedRows(ui->fits);
        for ( size_t i = 0; i < rows.size() && 2*i < clipboard.size(); i++ ) {
            getGraphColorBtn(rows[i])->setColor(clipboard[2*i]);
            getErrorColorBtn(rows[i])->setColor(clipboard[2*i+1]);
        }
        replot(false);
    });

    connect(ui->addGroup, SIGNAL(clicked(bool)), this, SLOT(addGroup()));
    connect(ui->delGroup, SIGNAL(clicked(bool)), this, SLOT(removeGroup()));

    connect(ui->fits, &QTableWidget::itemSelectionChanged, [=]() {
        QList<QTableWidgetSelectionRange> rlist = ui->fits->selectedRanges();
        ui->settingsButtons->setEnabled(rlist.size() == 1 && rlist.first().rowCount() == 1);
    });

    connect(ui->rundata, &QPushButton::clicked, [=](bool){
        std::vector<int> rows = getSelectedRows(ui->fits);
        if ( rows.size() != 1 )
            return;
        RunDataDialog *dlg = new RunDataDialog(this->session, this->session.gaFitter().results().at(rows[0]).resultIndex);
        dlg->setWindowTitle(QString("%1 for fit %2").arg(dlg->windowTitle()).arg(rows[0]));
        dlg->show();
    });
    connect(ui->daqdata, &QPushButton::clicked, [=](bool){
        std::vector<int> rows = getSelectedRows(ui->fits);
        if ( rows.size() != 1 )
            return;
        DAQDialog *dlg = new DAQDialog(this->session, this->session.gaFitter().results().at(rows[0]).resultIndex);
        dlg->setWindowTitle(QString("%1 for fit %2").arg(dlg->windowTitle()).arg(rows[0]));
        dlg->show();
    });
    connect(ui->fittersettings, &QPushButton::clicked, [=](bool){
        std::vector<int> rows = getSelectedRows(ui->fits);
        if ( rows.size() != 1 )
            return;
        GAFitterSettingsDialog *dlg = new GAFitterSettingsDialog(this->session, this->session.gaFitter().results().at(rows[0]).resultIndex);
        dlg->setWindowTitle(QString("%1 for fit %2").arg(dlg->windowTitle()).arg(rows[0]));
        dlg->show();
    });

    for ( int i = 0; i < 2; i++ ) {
        int c = ui->fits->columnCount();
        for ( const AdjustableParam &p : session.project.model().adjustableParams ) {
            ui->fits->insertColumn(c);
            QTableWidgetItem *item = new QTableWidgetItem(QString::fromStdString(p.name));
            item->setToolTip(i ? "Final value" : "Target value");
            ui->fits->setHorizontalHeaderItem(c, item);
            ui->fits->setColumnWidth(c, 70);
            ++c;
        }
        if ( !i ) {
            ui->fits->insertColumn(c);
            ui->fits->setHorizontalHeaderItem(c, new QTableWidgetItem(""));
            ui->fits->setColumnWidth(c, 10);
        }
    }
    ui->fits->setColumnWidth(0, 15);
    ui->fits->setColumnWidth(1, 15);
    ui->fits->setColumnWidth(3, 40);
    ui->fits->setColumnWidth(5, 40);
    ui->groups->setColumnWidth(1, 80);
    ui->groups->horizontalHeader()->setFixedHeight(5);
    ui->groups->verticalHeader()->setSectionsMovable(true);
    connect(&session.gaFitter(), SIGNAL(done()), this, SLOT(updateFits()));
    connect(ui->fits, &QTableWidget::itemSelectionChanged, [=]{
        replot(false);
    });
    connect(ui->groups, &QTableWidget::itemSelectionChanged, [=]{
        replot(true);
    });
    updateFits();

    connect(ui->plot_tabs, &QTabWidget::currentChanged, this, [=](){
        if ( ui->plot_tabs->currentWidget() == ui->fit_plots )
            ui->fit_plots->replot();
        else if ( ui->plot_tabs->currentWidget() == ui->deviation_boxplot )
            ui->deviation_boxplot->replot();
        else
            ui->error_plotter->replot();
    });

    ui->fit_plots->init(&session, false);
    ui->deviation_boxplot->init(&session);
    ui->error_plotter->init(&session);
}

FitInspector::~FitInspector()
{
    delete ui;
}

void FitInspector::replot(bool summarising)
{
    std::vector<Group> selection;
    std::vector<std::vector<int>> selected_groups;

    // Build Group vector
    if ( summarising ) {
        std::vector<int> selected_rows = getSelectedRows(ui->groups);
        selection.resize(selected_rows.size());
        for ( size_t i = 0; i < selected_rows.size(); i++ ) {
            int group = selected_rows[i];
            selection[i].label = ui->groups->item(group, 2)->text().isEmpty()
                    ? ui->groups->item(group, 1)->text()
                    : ui->groups->item(group, 2)->text();
            selection[i].color = getGroupColorBtn(group)->color;
            selected_groups.push_back(groups[group]);
        }
    } else {
        selected_groups.push_back(getSelectedRows(ui->fits));
        selection.resize(1);
        selection[0].color = QColor(Qt::black);
    }

    // Build Fit vector[s]
    for ( size_t i = 0; i < selected_groups.size(); i++ ) {
        selection[i].fits.resize(selected_groups[i].size());
        for ( size_t j = 0; j < selected_groups[i].size(); j++ ) {
            int fit = selected_groups[i][j];
            selection[i].fits[j].label = QString("Fit %1").arg(fit);
            selection[i].fits[j].color = getGraphColorBtn(fit)->color;
            selection[i].fits[j].errColor = getErrorColorBtn(fit)->color;
            selection[i].fits[j].idx = fit;
            selection[i].fits[j].gaf =& session.gaFitter();
        }
    }

    ui->fit_plots->setData(selection, summarising);
    ui->deviation_boxplot->setData(selection, summarising);
    ui->error_plotter->setData(selection, summarising);
}

ColorButton *FitInspector::getGraphColorBtn(int row)
{
    return qobject_cast<ColorButton*>(ui->fits->cellWidget(row, 0));
}

ColorButton *FitInspector::getErrorColorBtn(int row)
{
    return qobject_cast<ColorButton*>(ui->fits->cellWidget(row, 1));
}

ColorButton *FitInspector::getGroupColorBtn(int row)
{
    return qobject_cast<ColorButton*>(ui->groups->cellWidget(row, 0));
}

std::vector<int> FitInspector::getSelectedRows(QTableWidget* table)
{
    QList<QTableWidgetSelectionRange> selection = table->selectedRanges();
    std::vector<int> rows;
    for ( auto range : selection )
        for ( int i = range.topRow(); i <= range.bottomRow(); i++)
            rows.push_back(i);
    return rows;
}

void FitInspector::addGroup(std::vector<int> group, QString label)
{
    if ( group.empty() ) {
        group = getSelectedRows(ui->fits);
        if ( group.empty() )
            return;
        std::sort(group.begin(), group.end());
    }
    groups.push_back(group);

    int row = ui->groups->rowCount();
    ui->groups->insertRow(row);
    ColorButton *c = new ColorButton();
    c->setColor(QColorDialog::standardColor(row % 20));
    ui->groups->setCellWidget(row, 0, c);
    connect(c, &ColorButton::colorChanged, this, [=](){
        replot(true);
    });

    QString numbers;
    for ( int i = 0, last = group.size()-1; i <= last; i++ ) {
        int beginning = group[i];
        while ( i < last && group[i+1] == group[i] + 1 )
            ++i;
        if ( group[i] > beginning )
            numbers.append(QString("%1-%2").arg(beginning).arg(group[i]));
        else
            numbers.append(QString::number(group[i]));
        if ( i < last )
            numbers.append("; ");
    }
    ui->groups->setItem(row, 1, new QTableWidgetItem(numbers));

    QTableWidgetItem *item = new QTableWidgetItem(label);
    ui->groups->setItem(row, 2, item);
}

void FitInspector::removeGroup()
{
    std::vector<int> rows = getSelectedRows(ui->groups);
    std::sort(rows.begin(), rows.end(), [](int a, int b){return a > b;}); // descending
    for ( int row : rows ) {
        ui->groups->removeRow(row);
        groups.erase(groups.begin() + row);
    }
}

void FitInspector::on_saveGroups_clicked()
{
    if ( groups.empty() )
        return;
    QString file = QFileDialog::getSaveFileName(this, "Save groups to file...", session.directory());
    if ( file.isEmpty() )
        return;
    std::ofstream os(file.toStdString());
    for ( size_t vi = 0; vi < groups.size(); vi++ ) {
        int i = ui->groups->verticalHeader()->logicalIndex(vi);
        os << groups[i].size() << ':';
        for ( int f : groups[i] )
            os << f << ',';
        os << ui->groups->item(i, 2)->text().toStdString() << std::endl;
    }
}

void FitInspector::on_loadGroups_clicked()
{
    QString file = QFileDialog::getOpenFileName(this, "Select saved groups file...", session.directory());
    if ( file.isEmpty() )
        return;
    std::ifstream is(file.toStdString());
    int size;
    std::vector<int> group;
    char tmp;
    std::string label;
    is >> size;
    while ( is.good() ) {
        is >> tmp;
        group.resize(size);
        for ( int i = 0; i < size; i++ )
            is >> group[i] >> tmp;
        std::getline(is, label);
        addGroup(group, QString::fromStdString(label));
        is >> size;
    }
}

void FitInspector::updateFits()
{
    int col0 = 11;
    for ( size_t i = ui->fits->rowCount(); i < session.gaFitter().results().size(); i++ ) {
        const GAFitter::Output &fit = session.gaFitter().results().at(i);
        ui->fits->insertRow(i);
        ui->fits->setVerticalHeaderItem(i, new QTableWidgetItem(QString::number(i)));
        ColorButton *c = new ColorButton();
        c->setColor(ui->sepcols->isChecked() ? QColorDialog::standardColor(i%20) : Qt::blue);
        ui->fits->setCellWidget(i, 0, c);
        c = new ColorButton();
        c->setColor(ui->sepcols->isChecked() ? QColorDialog::standardColor(i%20 + 21) : Qt::magenta);
        ui->fits->setCellWidget(i, 1, c);
        ui->fits->setItem(i, 2, new QTableWidgetItem(fit.stimSource.prettyName()));
        ui->fits->setItem(i, 3, new QTableWidgetItem(QString::number(fit.epochs)));
        ui->fits->setItem(i, 4, new QTableWidgetItem(QString::number(session.gaFitterSettings(fit.resultIndex).randomOrder)));
        ui->fits->setItem(i, 5, new QTableWidgetItem(QString::number(session.gaFitterSettings(fit.resultIndex).crossover, 'g', 2)));
        ui->fits->setItem(i, 6, new QTableWidgetItem(session.gaFitterSettings(fit.resultIndex).decaySigma ? "Y" : "N"));
        ui->fits->setItem(i, 8, new QTableWidgetItem(session.gaFitterSettings(fit.resultIndex).useDE ? "DE" : "GA"));
        ui->fits->setItem(i, 9, new QTableWidgetItem(session.gaFitterSettings(fit.resultIndex).useClustering ? "Y" : "N"));
        ui->fits->setItem(i, 10, new QTableWidgetItem(QString::number(session.gaFitterSettings(fit.resultIndex).mutationSelectivity)));
        if ( session.daqData(fit.resultIndex).simulate == -1 ) {
            ui->fits->setItem(i, 7, new QTableWidgetItem(fit.VCRecord));
            for ( size_t j = 0; j < session.project.model().adjustableParams.size(); j++ )
                ui->fits->setItem(i, col0+j, new QTableWidgetItem(QString::number(fit.targets[j], 'g', 3)));
        } else if ( session.daqData(fit.resultIndex).simulate == 0 ) {
            ui->fits->setItem(i, 7, new QTableWidgetItem(QString("live DAQ")));
        } else if ( session.daqData(fit.resultIndex).simulate == 1 ) {
            ui->fits->setItem(i, 7, new QTableWidgetItem(QString("%1-%2")
                                                         .arg(session.daqData(fit.resultIndex).simulate)
                                                         .arg(session.daqData(fit.resultIndex).simd.paramSet)));
            for ( size_t j = 0; j < session.project.model().adjustableParams.size(); j++ )
                ui->fits->setItem(i, col0+j, new QTableWidgetItem(QString::number(fit.targets[j], 'g', 3)));
        }
        if ( fit.final )
            for ( size_t j = 0, np = session.project.model().adjustableParams.size(); j < np; j++ )
                ui->fits->setItem(i, col0+np+1+j, new QTableWidgetItem(QString::number(fit.finalParams[j], 'g', 3)));
    }
}
