#include "gafitterwidget.h"
#include "ui_gafitterwidget.h"
#include "cannedchannelassociationdialog.h"
#include "supportcode.h"

GAFitterWidget::GAFitterWidget(Session &session, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::GAFitterWidget),
    session(session),
    nQueued(0)
{
    ui->setupUi(this);

    ui->params_plotter->init(&session, true);

    connect(&session.wavesets(), SIGNAL(addedSet()), this, SLOT(updateDecks()));
    connect(&session.gaFitter(), SIGNAL(done()), this, SLOT(done()));
    connect(&session.gaFitter(), SIGNAL(progress(quint32)), this, SLOT(progress(quint32)));

    connect(&session.gaFitter(), SIGNAL(starting()), ui->response_plotter, SLOT(clear()));
    connect(&session.gaFitter(), SIGNAL(starting()), ui->response_plotter, SLOT(start()));
    connect(&session.gaFitter(), SIGNAL(done()), ui->response_plotter, SLOT(stop()));
    session.gaFitter().qV = &ui->response_plotter->qV;
    session.gaFitter().qI = &ui->response_plotter->qI;
    session.gaFitter().qO = &ui->response_plotter->qO;
    ui->response_plotter->VC =& session.runData().VC;

    connect(&session, &Session::DAQDataChanged, this, [=](){
        ui->records->setEnabled(this->session.qDaqData().simulate < 0);
    });
    ui->records->setEnabled(this->session.qDaqData().simulate < 0);

    for ( size_t i = 0; i < session.gaFitter().results().size(); i++ )
        ui->resumeSrc->addItem(QString("Fit %1 (%2)").arg(i).arg(session.gaFitter().results().at(i).resultIndex, 4, 10, QChar('0')));

    connect(&session, &Session::GAFitterSettingsChanged, this, &GAFitterWidget::updateDecks);
    updateDecks();

    connect(ui->finish, SIGNAL(clicked(bool)), &session.gaFitter(), SLOT(finish()), Qt::DirectConnection);
}

GAFitterWidget::~GAFitterWidget()
{
    delete ui;
}

void GAFitterWidget::updateDecks()
{
    ui->VCCreate->setEnabled(!session.wavesets().sources().empty());
    ui->start->setEnabled(!session.wavesets().sources().empty());

    int idx = ui->decks->currentIndex();
    ui->decks->clear();
    WaveSource empty(session, WaveSource::Empty, 0);
    ui->decks->addItem(empty.prettyName(), QVariant::fromValue(empty));
    for ( WaveSource &src : session.wavesets().sources() )
        ui->decks->addItem(src.prettyName(), QVariant::fromValue(src));
    ui->decks->setCurrentIndex(idx);
}

void GAFitterWidget::progress(quint32 idx)
{
    ui->label_epoch->setText(QString("Epoch %1/%2").arg(idx).arg(session.gaFitterSettings().maxEpochs));
}

void GAFitterWidget::done()
{
    nQueued--;
    ui->label_epoch->setText("");
    ui->label_queued->setText(QString("%1 queued").arg(nQueued));
    if ( nQueued > 0 )
        ui->params_plotter->clear();
    ui->resumeSrc->addItem(QString("Fit %1 (%2)").arg(session.gaFitter().results().size()-1).arg(session.gaFitter().results().back().resultIndex, 4, 10, QChar('0')));
    if ( ui->resumeSrc->currentIndex() == ui->resumeSrc->count() - 2 )
        ui->resumeSrc->setCurrentIndex(ui->resumeSrc->count() - 1);
}

void GAFitterWidget::unqueue(int n)
{
    nQueued += n;
    ui->label_queued->setText(QString("%1 queued").arg(nQueued));
}

void GAFitterWidget::on_start_clicked()
{
    if ( ui->decks->currentIndex() < 1 )
        return;
    if ( nQueued == 0 ) {
        ui->params_plotter->clear();
        ui->label_epoch->setText("Starting...");
    }
    WaveSource src = ui->decks->currentData().value<WaveSource>();

    if ( ui->resume->isChecked() ) {
        QStringList recs = ui->VCRecord->toPlainText().split('\n', QString::SkipEmptyParts);
        QString VCRecord = recs.isEmpty() ? QString() : recs.first();
        for ( int i = 0; i < ui->repeats->value(); i++ ) {
            session.gaFitter().resume(ui->resumeSrc->currentIndex(), src, VCRecord, ui->VCReadCfg->isChecked());
            ++nQueued;
        }
    } else if ( session.qDaqData().simulate == -1 ) {
        for ( const QString &record : ui->VCRecord->toPlainText().split('\n', QString::SkipEmptyParts) ) {
            for ( int i = 0; i < ui->repeats->value(); i++ ) {
                session.gaFitter().run(src, record.trimmed(), ui->VCReadCfg->isChecked());
                ++nQueued;
            }
        }
    } else {
        for ( int i = 0; i < ui->repeats->value(); i++ ) {
            session.gaFitter().run(src);
            ++nQueued;
        }
    }
    ui->label_queued->setText(QString("%1 queued").arg(nQueued));
}

void GAFitterWidget::on_abort_clicked()
{
    nQueued = 1;
    session.abort();
}

void GAFitterWidget::on_VCBrowse_clicked()
{
    QStringList items = ui->VCRecord->toPlainText().split('\n', QString::SkipEmptyParts);
    QFileDialog dlg(this, "Select voltage clamp recording(s)", items.isEmpty()?"":items.first(), "Recordings and file lists (*.atf *.recs)");
    dlg.setFileMode(QFileDialog::ExistingFiles);
    if ( !items.isEmpty() )
        dlg.selectFile(QString("\"") + items.join("\" \"") + "\"");
    if ( dlg.exec() ) {
        QString text = "";
        for ( const QString &fname : dlg.selectedFiles() ) {
            if ( fname.endsWith(".atf") ) {
                text += fname.trimmed() + "\n";
            } else {
                QFile file(fname);
                file.open(QIODevice::ReadOnly);
                QString contents = file.readAll();
                if ( !contents.trimmed().isEmpty() )
                    text.append(contents + "\n");
            }
        }
        ui->VCRecord->setPlainText(text.trimmed());
    }
}

void GAFitterWidget::on_VCChannels_clicked()
{
    QString record = ui->VCRecord->toPlainText().split('\n', QString::SkipEmptyParts).first();
    if ( !QFile(record).exists() || ui->decks->currentIndex() < 1 )
        return;
    CannedDAQ daq(session, session.qSettings());
    WaveSource src = ui->decks->currentData().value<WaveSource>();
    daq.setRecord(src.stimulations(), record, false);
    CannedChannelAssociationDialog *dlg = new CannedChannelAssociationDialog(session, &daq, this);
    dlg->open();
}

void GAFitterWidget::on_VCCreate_clicked()
{
    if ( ui->decks->currentIndex() < 1 )
        return;
    QString file = QFileDialog::getSaveFileName(this, "Select output file");
    if ( file.isEmpty() )
        return;
    if ( !file.endsWith(".atf") )
        file.append(".atf");

    std::ofstream of(file.toStdString());
    if ( !of.good() )
        return;
    WaveSource src = ui->decks->currentData().value<WaveSource>();
    std::vector<Stimulation> stims = src.stimulations();

    int nTotal, nSamples, nBuffer;
    double dt = session.qRunData().dt;
    if ( session.qDaqData().filter.active )
        dt /= session.qDaqData().filter.samplesPerDt;
    CannedDAQ(session, session.qSettings()).getSampleNumbers(stims, dt, &nTotal, &nBuffer, &nSamples);

    // For details on the ATF format used as a stimulation file, see http://mdc.custhelp.com/app/answers/detail/a_id/17029
    // Note, the time column's values are ignored for stimulation.
    of << "ATF\t1.0\r\n";
    of << "2\t" << (stims.size()+1) << "\r\n";
    of << "\"Comment=RTDO VC stimulations. Project " << QDir(session.project.dir()).dirName() << " (" << session.project.model().name()
       << "), Session " << session.name() << ", " << src.prettyName() << "\"\r\n";
    of << "\"Use=Use as stimulation file with sampling interval " << (1000*dt) << " us (" << (1/dt) << " kHz) and "
       << nTotal << " total samples (" << (nTotal*dt/1000) << " s including buffers).\"\r\n";
    of << "\"Time (s)\"";
        for ( size_t i = 0; i < stims.size(); i++ ) {
            if ( src.type == WaveSource::Deck )
                of << "\t\"" << session.project.model().adjustableParams[i].name << " (mV)\"";
            else
                of << "\t\"stim " << i << " (mV)\"";
        }
        of << "\r\n";

    for ( int i = 0; i < nBuffer; i++ ) {
        of << (i - nBuffer) * dt * 1e-3;
        for ( const Stimulation &stim : stims )
            of << '\t' << stim.baseV;
        of << "\r\n";
    }

    for ( int i = 0; i < nSamples; i++ ) {
        double t = i * dt;
        of << t * 1e-3;
        for ( const Stimulation &stim : stims )
            of << '\t' << getCommandVoltage(stim, t);
        of << "\r\n";
    }

    for ( int i = 0; i < nBuffer; i++ ) {
        of << (nSamples + i) * dt * 1e-3;
        for ( const Stimulation &stim : stims )
            of << '\t' << stim.baseV;
        of << "\r\n";
    }

    std::ofstream meta(file.append(".stims").toStdString());
    if ( !meta.good() )
        return;
    WavesetCreator::writeStims(stims, meta, dt);
    meta << "\n\n######\n\n";
    for ( const Stimulation &I : stims )
        meta << I << std::endl;
}

void GAFitterWidget::on_cl_run_clicked()
{
    WaveSource src;
    if ( ui->decks->currentIndex() < 0 )
        src = ui->decks->itemData(0).value<WaveSource>();
    else
        src = ui->decks->currentData().value<WaveSource>();

    if ( ui->resume->isChecked() )
        session.gaFitter().cl_resume(ui->resumeSrc->currentIndex(), src);
    else
        session.gaFitter().cl_run(src);

    ++nQueued;
    ui->label_queued->setText(QString("%1 queued").arg(nQueued));
}

void GAFitterWidget::on_validate_clicked()
{
    if ( ui->resumeSrc->currentIndex() < 0 )
        return;
    session.gaFitter().validate(ui->resumeSrc->currentIndex());
}
