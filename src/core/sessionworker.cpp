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
