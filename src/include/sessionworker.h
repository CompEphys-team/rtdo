#ifndef SESSIONWORKER_H
#define SESSIONWORKER_H

#include <QObject>
#include <QFile>
#include "types.h"

class SessionWorker : public QObject
{
    Q_OBJECT
public:
    SessionWorker(Session &session);

    virtual inline void abort() {}

    Session &session;

protected:
    friend class Session;
    virtual void load(const QString &action, const QString &args, QFile &results, Result r) = 0;
    virtual QString actorName() const = 0;


    /**
     * @brief openSaveStream creates a file for binary results output. @sa openLoadStream()
     * @param file is the file to be created (or truncated).
     * @param os is populated with a ready-to-write data stream.
     * @param format_magic is an optional "magic number" to indicate file format. It is output as the first object in the stream.
     * @param version is an optional format version number, which is output in second place.
     * @return true on success, false if the file could not be opened.
     */
    bool openSaveStream(QFile &file, QDataStream &os, quint32 format_magic = 0, quint32 version = 0);

    /**
     * @brief openLoadStream opens a file for binary results loading. Throws a std::runtime_error on failure. @sa openSaveStream()
     * @param file is the file to be opened for reading.
     * @param is is populated with a ready-to-read data stream.
     * @param format_magic is an optional "magic number" to indicate file format. If non-zero, this will be checked against the first
     * quint32 in the file. An exception is thrown on mismatch.
     * @return the version number indicated in the file.
     */
    quint32 openLoadStream(QFile &file, QDataStream &is, quint32 format_magic = 0);
};

#endif // SESSIONWORKER_H
