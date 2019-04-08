#include "gafitter.h"
#include "session.h"
#include "daqfilter.h"

const QString GAFitter::action = QString("fit");
const quint32 GAFitter::magic = 0xadb8d269;
const quint32 GAFitter::version = 108;

GAFitter::GAFitter(Session &session) :
    SessionWorker(session),
    lib(session.project.universal()),
    settings(session.gaFitterSettings()),
    qV(nullptr),
    qI(nullptr),
    qO(nullptr)
{
}

GAFitter::~GAFitter()
{
}

GAFitter::Output::Output(Session &s, WaveSource stimSource, QString VCRecord, Result r) :
    Result(r),
    params(s.qGaFitterSettings().maxEpochs, std::vector<scalar>(s.project.model().adjustableParams.size())),
    error(s.qGaFitterSettings().maxEpochs),
    targetStim(s.qGaFitterSettings().maxEpochs),
    targets(s.project.model().adjustableParams.size()),
    stimSource(stimSource),
    variance(0),
    VCRecord(VCRecord),
    assoc(s.cdaq_assoc),
    finalParams(s.project.model().adjustableParams.size()),
    finalError(s.project.model().adjustableParams.size())
{
    this->stimSource.session =& s;
}

void GAFitter::run(WaveSource src, QString VCRecord, bool readRecConfig)
{
    if ( readRecConfig && session.qDaqData().simulate == -1 ) {
        QString cfg = VCRecord;
        cfg.replace(".atf", ".cfg");
        if ( QFileInfo(cfg).exists() )
            session.loadConfig(cfg);
    }
    session.queue(actorName(), action, src.prettyName(), new Output(session, src, VCRecord));
}

void GAFitter::resume(size_t fitIdx, WaveSource src, QString VCRecord, bool readRecConfig)
{
    if ( fitIdx >= results().size() )
        return;
    if ( readRecConfig && session.qDaqData().simulate == -1 ) {
        QString cfg = VCRecord;
        cfg.replace(".atf", ".cfg");
        if ( QFileInfo(cfg).exists() )
            session.loadConfig(cfg);
    }
    Output *out = new Output(m_results[fitIdx]);
    out->stimSource = src;
    out->VCRecord = VCRecord;
    session.queue(actorName(), action, src.prettyName(), out);
}

bool GAFitter::execute(QString action, QString, Result *res, QFile &file)
{
    if ( action != this->action )
        return false;

    {
        QMutexLocker locker(&mutex);
        doFinish = false;
        aborted = false;
    }

    output = std::move(*static_cast<Output*>(res));
    delete res;

    if ( output.stimSource.type != WaveSource::Deck && settings.mutationSelectivity == 2 ) {
        std::cerr << "Error: Non-deck sources must not use target-only mutation selectivity." << std::endl;
        return false;
    }

    if ( output.dryrun ) {
        save(file);
        return true;
    }

    emit starting();

    QTime wallclock = QTime::currentTime();
    double simtime = 0;
    qT = 0;

    astims = output.stimSource.stimulations();

    daq = new DAQFilter(session, session.getSettings());

    if ( session.daqData().simulate < 0 ) {
        daq->getCannedDAQ()->assoc = output.assoc;
        daq->getCannedDAQ()->setRecord(astims, output.VCRecord);
        output.variance = daq->getCannedDAQ()->variance;
        std::cout << "Baseline current noise s.d.: " << std::sqrt(output.variance) << " nA" << std::endl;
    }

    if ( session.daqData().simulate == 0 )
        for ( size_t i = 0; i < lib.adjustableParams.size(); i++ )
            output.targets[i] = lib.adjustableParams[i].initial;
    else
        for ( size_t i = 0; i < lib.adjustableParams.size(); i++ )
            output.targets[i] = daq->getAdjustableParam(i);

    setup();

    populate();

    simtime = fit();

    // Finalise ranking and select winning parameter set
    simtime += finalise();

    double wallclockms = wallclock.msecsTo(QTime::currentTime());
    std::cout << "ms elapsed: " << wallclockms << std::endl;
    std::cout << "ms simulated: " << simtime << std::endl;
    std::cout << "Ratio: " << (simtime/wallclockms) << std::endl;

    // Resumability
    output.resume.population.assign(lib.adjustableParams.size(), std::vector<scalar>(lib.NMODELS));
    for ( size_t i = 0; i < lib.adjustableParams.size(); i++ )
        for ( size_t j = 0; j < lib.NMODELS; j++ )
            output.resume.population[i][j] = lib.adjustableParams[i][j];
    output.resume.bias = bias;
    output.resume.DEMethodUsed = DEMethodUsed;
    output.resume.DEMethodSuccess = DEMethodSuccess;
    output.resume.DEMethodFailed = DEMethodFailed;
    output.resume.DEpX = DEpX;

    // Finish
    output.epochs = epoch;
    m_results.push_back(std::move(output));
    emit done();

    // Save
    save(file);

    delete daq;
    daq = nullptr;

    return true;
}

void GAFitter::save(QFile &file)
{
    QDataStream os;
    if ( openSaveStream(file, os, magic, version) ) {
        const Output &out = m_results.back();
        os << out.stimSource << out.epochs;
        for ( quint32 i = 0; i < out.epochs; i++ ) {
            os << out.targetStim[i] << out.error[i];
            for ( const scalar &p : out.params[i] )
                os << p;
        }
        for ( const scalar &t : out.targets )
            os << t;
        os << out.final;
        for ( const scalar &p : out.finalParams )
            os << p;
        for ( const scalar &e : out.finalError )
            os << e;
        os << out.VCRecord;
        os << out.variance;
        os << out.stims;
        os << out.obs;
        os << out.baseF;
        os << qint32(out.assoc.Iidx) << qint32(out.assoc.Vidx) << qint32(out.assoc.V2idx);
        os << out.assoc.Iscale << out.assoc.Vscale << out.assoc.V2scale;

        for ( const auto &vec : out.resume.population )
            for ( const scalar &v : vec )
                os << v;
        for ( const double &b : out.resume.bias )
            os << b;
        for ( const int &n : out.resume.DEMethodUsed )
            os << n;
        for ( const int &n : out.resume.DEMethodSuccess )
            os << n;
        for ( const int &n : out.resume.DEMethodFailed )
            os << n;
        for ( const double &p : out.resume.DEpX )
            os << p;
    }
}

void GAFitter::finish()
{
    QMutexLocker locker(&mutex);
    doFinish = true;
}

Result *GAFitter::load(const QString &act, const QString &, QFile &results, Result r)
{
    if ( act != action )
        throw std::runtime_error(std::string("Unknown action: ") + act.toStdString());
    QDataStream is;
    quint32 ver = openLoadStream(results, is, magic);
    if ( ver < 100 || ver > version )
        throw std::runtime_error(std::string("File version mismatch: ") + results.fileName().toStdString());

    Output *p_out;
    if ( r.dryrun ) {
        p_out = new Output(session, WaveSource(), QString(), r);
    } else {
        m_results.emplace_back(session, WaveSource(), QString(), r);
        p_out =& m_results.back();
    }
    Output &out = *p_out;

    is >> out.stimSource >> out.epochs;
    out.targetStim.resize(out.epochs);
    out.params.resize(out.epochs);
    out.error.resize(out.epochs);
    for ( quint32 i = 0; i < out.epochs; i++ ) {
        is >> out.targetStim[i] >> out.error[i];
        for ( scalar &p : out.params[i] )
            is >> p;
    }
    if ( ver >= 101 )
        for ( scalar &t : out.targets )
            is >> t;
    if ( ver >= 102 ) {
        is >> out.final;
        for ( scalar &p : out.finalParams )
            is >> p;
        for ( scalar &e : out.finalError )
            is >> e;
    }
    if ( ver >= 103 ) {
        is >> out.VCRecord;
        if ( session.daqData().simulate < 0 ) {
            CannedDAQ tmp(session, session.getSettings());
            tmp.setRecord(out.stimSource.stimulations(), out.VCRecord, false);
            for ( size_t i = 0; i < lib.adjustableParams.size(); i++ )
                if ( settings.constraints[i] != 3 ) // Never overwrite fixed target
                    out.targets[i] = tmp.getAdjustableParam(i);
        }
    }
    if ( ver >= 104 )
        is >> out.variance;
    if ( ver >= 105 ) {
        if ( ver < 107 ) {
            QVector<Stimulation> stims;
            QVector<QVector<QPair<int,int>>> obsTimes;
            is >> stims >> obsTimes;
            for ( const Stimulation &I : stims )
                out.stims.push_back(iStimulation(I, session.runData().dt));
            for ( QVector<QPair<int,int>> obst : obsTimes ) {
                out.obs.push_back({{},{}});
                for ( int i = 0; i < obst.size(); i++ ) {
                    if ( i < int(iObservations::maxObs) ) {
                        out.obs.back().start[i] = obst[i].first;
                        out.obs.back().stop[i] = obst[i].second;
                    }
                }
            }
        } else {
            is >> out.stims;
            is >> out.obs;
        }
        is >> out.baseF;
    }
    if ( ver >= 106 ) {
        qint32 I, V, V2;
        is >> I >> V >> V2;
        out.assoc.Iidx = I;
        out.assoc.Vidx = V;
        out.assoc.V2idx = V2;
        is >> out.assoc.Iscale >> out.assoc.Vscale >> out.assoc.V2scale;
    }

    if ( ver >= 108 ) {
        size_t nParams = lib.adjustableParams.size();
        out.resume.population.assign(nParams, std::vector<scalar>(lib.NMODELS));
        out.resume.bias.resize(out.stims.size());
        out.resume.DEMethodUsed.resize(nParams);
        out.resume.DEMethodSuccess.resize(nParams);
        out.resume.DEMethodFailed.resize(nParams);
        out.resume.DEpX.resize(nParams);

        for ( auto &vec : out.resume.population )
            for ( scalar &v : vec )
                is >> v;
        for ( double &b : out.resume.bias )
            is >> b;
        for ( int &n : out.resume.DEMethodUsed )
            is >> n;
        for ( int &n : out.resume.DEMethodSuccess )
            is >> n;
        for ( int &n : out.resume.DEMethodFailed )
            is >> n;
        for ( double &p : out.resume.DEpX )
            is >> p;
    }

    return p_out;
}

bool GAFitter::finished()
{
    QMutexLocker locker(&mutex);
    return aborted || doFinish || epoch >= settings.maxEpochs;
}

bool GAFitter::errTupelSort(const errTupel &x, const errTupel &y)
{
    // make sure NaN is largest
    if ( std::isnan(x.err) ) return false;
    if ( std::isnan(y.err) ) return true;
    return x.err < y.err;
}

void GAFitter::pushToQ(double t, double V, double I, double O)
{
    if ( qV )
        qV->push({t,V});
    if ( qI )
        qI->push({t,I});
    if ( qO )
        qO->push({t,O});
}
