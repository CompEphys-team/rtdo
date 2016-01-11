#include "util.h"

QString dirname(QString path) {
    return dirname(path.toStdString());
}

QString dirname(std::string path) {
    int lastslash = path.find_last_of('/');
    if ( lastslash )
        return QString::fromStdString(path.substr(0, lastslash));
    else
        return QString();
}
