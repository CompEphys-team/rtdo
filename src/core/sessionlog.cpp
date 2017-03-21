#include "sessionlog.h"
#include <QFileInfo>
#include <cassert>
#include "streamops.h"

SessionLog::SessionLog()
{

}

void SessionLog::setLogFile(const QString &path)
{
    if ( QFileInfo(path).exists() ) {
        std::ifstream is(path.toStdString());
        while ( is.good() && is.peek() != EOF ) {
            std::string line;
            std::getline(is, line);
            QString qline = QString::fromStdString(line);
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

int SessionLog::put(const QString &actor, const QString &action, const QString &args)
{
    // Actor and Action must not be empty
    assert(!actor.isEmpty() && !action.isEmpty());

    int idx = m_data.size();
    Entry entry {QDateTime::currentDateTime(), actor, action, args};

    // Insert new entry into model
    beginInsertRows(QModelIndex(), m_data.size(), 1);
    m_data.push_back(entry);
    endInsertRows();

    // Write to log file
    if ( m_file.is_open() )
        m_file << data(index(idx, 0), Qt::UserRole).toString() << std::endl;

    return idx;
}

QVariant SessionLog::data(const QModelIndex &index, int role) const
{
    if ( index.row() < 0 || index.row() >= rowCount() )
        return QVariant();
    // Table mode, display one item at a time. See also shortcut functions timestamp(), actor(), action(), args().
    if ( role == Qt::DisplayRole ) {
        switch ( index.column() ) {
        case 0: return m_data.at(index.row()).timestamp;
        case 1: return m_data.at(index.row()).actor;
        case 2: return m_data.at(index.row()).action;
        case 3: return m_data.at(index.row()).args;
        }
    // Log line mode, display the whole shebang
    } else if ( role == Qt::UserRole ) {
        const Entry &e = m_data.at(index.row());
        return QString("%1\t%2\t%3\t%4")
                .arg(e.timestamp.toString(Qt::ISODate))
                .arg(e.actor, e.action, e.args);
    }
    return QVariant();
}

SessionLog::Entry SessionLog::entry(int row) const
{
    if ( row < 0 || row >= rowCount() )
        return Entry();
    return m_data.at(row);
}
