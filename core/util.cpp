/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-20

--------------------------------------------------------------------------*/
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
