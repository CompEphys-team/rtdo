#ifndef SESSIONLOG_H
#define SESSIONLOG_H

#include <QAbstractTableModel>
#include <QDateTime>
#include <fstream>

class SessionLog : public QAbstractTableModel
{
public:
    SessionLog();

    void setLogFile(const QString &path);

    int put(const QString &actor, const QString &action, const QString &args);

    QVariant data(const QModelIndex &index, int role) const;

    inline int columnCount(const QModelIndex & = QModelIndex()) const { return 4; }
    inline int rowCount(const QModelIndex & = QModelIndex()) const { return m_data.size(); }

protected:
    struct Entry {
        QDateTime timestamp;
        QString actor;
        QString action;
        QString args;
    };

    QVector<Entry> m_data;

    std::ofstream m_file;
};

#endif // SESSIONLOG_H
