/** RTDO: Closed loop neuron model fitting
 *  Copyright (C) 2019 Felix Kern <kernfel+github@gmail.com>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
**/


#ifndef SESSIONWORKER_H
#define SESSIONWORKER_H

#include <QObject>
#include <QFile>
#include <QMutex>
#include "types.h"

class SessionWorker : public QObject
{
    Q_OBJECT
public:
    SessionWorker(Session &session);

    ///! Abort the currently executing action. Worker is responsible for resetting aborted through clearAbort().
    virtual inline void abort() { QMutexLocker locker(&mutex); aborted = true; }

    Session &session;

    virtual QString actorName() const = 0;
    virtual bool execute(QString action, QString args, Result *res, QFile &file) = 0;

protected:
    friend class Session;
    virtual Result *load(const QString &action, const QString &args, QFile &results, Result r) = 0;

    bool aborted = false;
    mutable QMutex mutex;

    inline bool isAborted() { QMutexLocker locker(&mutex); return aborted; }
    inline void clearAbort() { QMutexLocker locker(&mutex); aborted = false; }

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
