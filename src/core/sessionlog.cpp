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


#include "sessionlog.h"
#include <QFileInfo>
#include <cassert>
#include <QBrush>
#include "streamops.h"

SessionLog::SessionLog()
{

}

SessionLog::~SessionLog()
{
    while ( !m_queue.isEmpty() )
        delete m_queue.dequeue().res;
}

void SessionLog::setLogFile(const QString &path)
{
    if ( QFileInfo(path).exists() ) {
        std::ifstream is(path.toStdString());
        while ( is.good() && is.peek() != EOF ) {
            std::string line;
            std::getline(is, line);
            QString qline = QString::fromStdString(line);
            if ( qline.startsWith('#') )
                continue;
            QStringList list = qline.split(QChar('\t'));
            m_data.push_back({
                 QDateTime::fromString(list[0], Qt::ISODate),
                 list[1],
                 list[2],
                 list.value(3)
             });
        }
    }
    m_file.open(path.toStdString(), std::ios_base::out | std::ios_base::app);
}

void SessionLog::setDesiccateFile(const QString &path, const QString &dopfile, const QString &sessdir)
{
    if ( !path.isEmpty() ) {
        m_sicfile.open(path.toStdString(), std::ios_base::out | std::ios_base::trunc);
        m_sicfile << "#project " << dopfile << '\n'
                  << "#session " << sessdir << std::endl;
        nDesiccatedActions = 0;
    }
}

void SessionLog::clearDesiccateFile()
{
    if ( m_sicfile.is_open() ) {
        m_sicfile.close();
        beginRemoveRows(QModelIndex(), rowCount()-nDesiccatedActions, rowCount()-1);
        m_data.erase(m_data.end()-nDesiccatedActions, m_data.end());
        endRemoveRows();
    }
}

void SessionLog::queue(QString actor, QString action, QString args, Result *res)
{
    queue(Entry {QDateTime::currentDateTime(), actor, action, args, res});
}

void SessionLog::queue(const SessionLog::Entry &entry)
{
    beginInsertRows(QModelIndex(), rowCount(), rowCount());
    m_queue.enqueue(entry);
    endInsertRows();
}

SessionLog::Entry SessionLog::dequeue(bool makeActive)
{
    if ( makeActive ) {
        if ( m_hasActive )
            clearActive(false);
        m_hasActive = true;
        m_active = m_queue.dequeue();
        m_active.timestamp = QDateTime::currentDateTime();
        emit dataChanged(index(m_data.size(), 0), index(m_data.size(), columnCount()));
        return m_active;
    } else {
        int row = m_data.size() + m_hasActive;
        beginRemoveRows(QModelIndex(), row, row);
        Entry ret = m_queue.dequeue();
        endRemoveRows();
        return ret;
    }
}

void SessionLog::removeQueued(int first, int last)
{
    int qStart = m_data.size() + m_hasActive;
    if ( first < qStart )
        first = qStart;
    if ( last > rowCount() )
        last = rowCount();
    beginRemoveRows(QModelIndex(), first, last);
    for ( int i = first; i <= last; i++ ) {
        delete m_queue.takeAt(first - qStart).res;
    }
    endRemoveRows();
    emit queueAltered();
}

void SessionLog::clearActive(bool success)
{
    if ( !m_hasActive )
        return;
    int row = m_data.size();
    if ( success ) {
        m_active.timestamp = QDateTime::currentDateTime();
        m_active.res = nullptr;
        m_data.push_back(m_active);
        m_hasActive = false;
        emit dataChanged(index(row, 0), index(row, columnCount()));

        // Write to log file
        if ( m_sicfile.is_open() ) {
            m_sicfile << data(index(row, 0), Qt::UserRole).toString() << std::endl;
            ++nDesiccatedActions;
        } else if ( m_file.is_open() )
            m_file << data(index(row, 0), Qt::UserRole).toString() << std::endl;
    } else {
        beginRemoveRows(QModelIndex(), row, row);
        m_hasActive = false;
        endRemoveRows();
    }
}

QVariant SessionLog::headerData(int section, Qt::Orientation orientation, int role) const
{
    if ( orientation == Qt::Vertical ) {
        if ( role == Qt::DisplayRole )
            return QString("%1").arg(section, 4, 10, QChar('0'));
    } else {
        if ( role == Qt::DisplayRole ) {
            switch ( section ) {
            case 0: return QString("Time");
            case 1: return QString("Actor");
            case 2: return QString("Action");
            case 3: return QString("Arguments");
            }
        }
    }
    return QVariant();
}

QVariant SessionLog::data(const QModelIndex &index, int role) const
{
    if ( index.row() < 0 || index.row() >= rowCount() )
        return QVariant();

    Entry e = entry(index.row());

    // Table mode, display one item at a time.
    if ( role == Qt::DisplayRole ) {
        switch ( index.column() ) {
        case 0: return e.timestamp;
        case 1: return e.actor;
        case 2: return e.action;
        case 3: return e.args;
        }
    // Log line mode, display the whole shebang
    } else if ( role == Qt::UserRole ) {
        return QString("%1\t%2\t%3\t%4")
                .arg(e.timestamp.toString(Qt::ISODate))
                .arg(e.actor, e.action, e.args);
    } else if ( role == Qt::BackgroundRole ) {
        if ( index.row() < m_data.size() )
            return QBrush(QColor(220,220,220));
        else if ( index.row() == m_data.size() && m_hasActive )
            return QBrush(QColor(128,255,128));
        else
            return QBrush(QColor(179,209,255));
    }
    return QVariant();
}

SessionLog::Entry SessionLog::entry(int row) const
{
    if ( row < 0 || row >= rowCount() )
        return Entry();
    else if ( row < m_data.size() )
        return m_data.at(row);
    else if ( m_hasActive && row == m_data.size() )
        return m_active;
    else
        return m_queue.at(row - m_data.size() - m_hasActive);
}
