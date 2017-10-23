#include "sessionworker.h"
#include <QDataStream>
#include "session.h"

SessionWorker::SessionWorker(Session &session) :
    session(session)
{
}

bool SessionWorker::openSaveStream(QFile &file, QDataStream &os, quint32 format_magic, quint32 version)
{
    if ( !file.open(QIODevice::WriteOnly) )
        return false;
    os.setDevice(&file);
    os << format_magic << version;
    os.setVersion(QDataStream::Qt_5_7);
    return true;
}

quint32 SessionWorker::openLoadStream(QFile &file, QDataStream &is, quint32 format_magic)
{
    if ( !file.open(QIODevice::ReadOnly) )
        throw std::runtime_error(std::string("Failed to open file ") + file.fileName().toStdString() + " for reading.");
    is.setDevice(&file);
    quint32 magic, version;
    is >> magic;
    if ( format_magic && magic != format_magic )
        throw std::runtime_error(std::string("File format mismatch in file ") + file.fileName().toStdString());
    is >> version;
    is.setVersion(QDataStream::Qt_5_7);
    return version;
}
