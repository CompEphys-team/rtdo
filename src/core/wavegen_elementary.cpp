#include "wavegen.h"
#include "session.h"

QString Wavegen::ee_action = QString("ee_search");
quint32 Wavegen::ee_magic = 0xc9fd545f;
quint32 Wavegen::ee_version = 100;

void Wavegen::ee_save(QFile &file)
{
    QDataStream os;
    if ( !openSaveStream(file, os, ee_magic, ee_version) )
        return;
    Archive &arch = m_archives.back();
    os << quint32(arch.precision);
    os << quint32(arch.iterations);
    os << arch.nCandidates << arch.nInsertions << arch.nReplacements << arch.nElites;
    os << arch.deltabar;

    os << quint32(arch.elites.size());
    os << quint32(arch.elites.front().bin.size());

    // Separate waves into unique and shared to reduce the number of lookups
    std::vector<iStimulation*> w_unique, w_shared;
    for ( MAPElite const& e : arch.elites ) {
        os << e.fitness;
        for ( size_t b : e.bin )
            os << quint32(b);

        if ( e.wave.unique() ) {
            w_unique.push_back(e.wave.get());
            os << qint32(-w_unique.size());
        } else {
            size_t i;
            for ( i = 0; i < w_shared.size(); i++ )
                if ( e.wave.get() == w_shared[i] )
                    break;
            if ( i == w_shared.size() )
                w_shared.push_back(e.wave.get());
            os << qint32(i);
        }
    }
    os << quint32(w_unique.size());
    for ( const iStimulation *pstim : w_unique )
        os << *pstim;
    os << quint32(w_shared.size());
    for ( const iStimulation *pstim : w_shared )
        os << *pstim;
}

void Wavegen::ee_load(QFile &file, const QString &, Result r)
{
    QDataStream is;
    quint32 version = openLoadStream(file, is, ee_magic);
    if ( version < 100 || version > ee_version )
        throw std::runtime_error(std::string("File version mismatch: ") + file.fileName().toStdString());

    m_archives.emplace_back(-1, searchd, r);
    Archive &arch = m_archives.back();

    quint32 precision, iterations, archSize, nBins;
    is >> precision >> iterations;
    arch.precision = precision;
    arch.iterations = iterations;
    is >> arch.nCandidates >> arch.nInsertions >> arch.nReplacements >> arch.nElites;
    is >> arch.deltabar;

    is >> archSize >> nBins;
    arch.elites.resize(archSize);
    std::vector<qint32> stimIdx(archSize);
    auto idxIt = stimIdx.begin();
    quint32 tmp;
    for ( auto el = arch.elites.begin(); el != arch.elites.end(); el++, idxIt++ ) {
        is >> el->fitness;
        el->bin.resize(nBins);
        for ( size_t &b : el->bin ) {
            is >> tmp;
            b = size_t(tmp);
        }
        is >> *idxIt;
    }

    quint32 uniqueSize, sharedSize;
    is >> uniqueSize;
    std::vector<std::shared_ptr<iStimulation>> w_unique(uniqueSize);
    for ( std::shared_ptr<iStimulation> &ptr : w_unique ) {
        ptr.reset(new iStimulation);
        is >> *ptr;
    }
    is >> sharedSize;
    std::vector<std::shared_ptr<iStimulation>> w_shared(sharedSize);
    for ( std::shared_ptr<iStimulation> &ptr : w_shared ) {
        ptr.reset(new iStimulation);
        is >> *ptr;
    }

    idxIt = stimIdx.begin();
    for ( auto el = arch.elites.begin(); el != arch.elites.end(); el++, idxIt++ ) {
        if ( *idxIt < 0 )
            el->wave = w_unique[-1 - *idxIt];
        else
            el->wave = w_shared[*idxIt];
    }
}
