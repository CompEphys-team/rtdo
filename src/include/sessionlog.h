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


#ifndef SESSIONLOG_H
#define SESSIONLOG_H

#include <QAbstractTableModel>
#include <QDateTime>
#include <QQueue>
#include <fstream>
#include "types.h"

class SessionLog : public QAbstractTableModel
{
    Q_OBJECT
public:
    SessionLog();
    ~SessionLog();

    struct Entry {
        Entry() {}
        Entry(QDateTime t, QString actor, QString action, QString args, Result *r = nullptr) :
            timestamp(std::move(t)),
            actor(std::move(actor)),
            action(std::move(action)),
            args(std::move(args)),
            res(r)
        {}
        QDateTime timestamp;
        QString actor;
        QString action;
        QString args;
        Result *res = nullptr;
    };

    void setLogFile(const QString &path);
    void setDesiccateFile(const QString &path, const QString &dopfile, const QString &sessdir);
    void clearDesiccateFile();

    int nextIndex() const { return m_data.size(); }

    void queue(QString actor, QString action, QString args, Result *res);
    void queue(const Entry &entry);
    int queueSize() const { return m_queue.size(); }
    Entry dequeue(bool makeActive);
    void removeQueued(int first, int last);

    void clearActive(bool success);

    QVariant headerData(int section, Qt::Orientation orientation, int role) const;
    QVariant data(const QModelIndex &index, int role) const;
    Entry entry(int row) const;

    inline int columnCount(const QModelIndex & = QModelIndex()) const { return 4; }
    inline int rowCount(const QModelIndex & = QModelIndex()) const { return m_data.size() + m_hasActive + m_queue.size(); }

signals:
    void queueAltered();

protected:
    QVector<Entry> m_data;
    QQueue<Entry> m_queue;
    Entry m_active;
    bool m_hasActive = false;

    std::ofstream m_file;
    std::ofstream m_sicfile;
    int nDesiccatedActions;
};

#endif // SESSIONLOG_H
