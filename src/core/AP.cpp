#include "AP.h"

#define RTDO_PROTOCOL_VERSION 1
#define RTDO_PROTOCOL_HEADER "#rtdo_config_version"
int LOADED_PROTOCOL_VERSION;

std::istream &operator>>(std::istream &is, QString &str)
{
    std::string tmp;
    is >> tmp;
    str = QString::fromStdString(tmp);
    return is;
}

std::ostream &operator<<(std::ostream &os, const QString &str)
{
    os << str.toStdString();
    return os;
}

void writeProtocol(std::ostream &os)
{
    os << RTDO_PROTOCOL_HEADER << " " << RTDO_PROTOCOL_VERSION << std::endl << std::endl;

    for ( auto const& ap : AP::params() ) {
        ap->write(os);
    }
}

bool readProtocol(std::istream &is, std::function<bool(QString)> *callback)
{
    QString name, header;
    int version = 0;
    std::vector<std::unique_ptr<AP>> deprec;

    if ( is.good() ) {
        is >> header;
        if ( is.good() && !header.isEmpty() && header == RTDO_PROTOCOL_HEADER ) {
            is >> version;
        } else {
            return false;
        }
    } else {
        return false;
    }

    LOADED_PROTOCOL_VERSION = version;

    AP *it;
    is >> name;
    while ( is.good() ) {
        bool ok = false;
        if ( (it = AP::find(name)) ) {
            it->readNow(name, is, &ok);
        } else if ( version < RTDO_PROTOCOL_VERSION ) {
            if ( (it = AP::find(name, &deprec)) )
                it->readNow(name, is, &ok);
        }
        if ( !ok && callback )
            if ( (*callback)(name) )
                return false;
        is >> name;
    }

    return true;
}
