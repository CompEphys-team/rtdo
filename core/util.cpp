/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-20

--------------------------------------------------------------------------*/
#include "util.h"
#include "shared.h"

using namespace std;

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

string basename_nosuffix(const string &path) {
    int lastslash = path.find_last_of('/');
    int lastperiod = path.find_last_of('.');
    if ( lastslash && lastslash+1 < lastperiod ) {
        return path.substr(lastslash+1, lastperiod-lastslash-1);
    } else {
        return string();
    }
}

ostream &operator<<(ostream &os, inputSpec &I) {
    os << I.t << " " << I.ot << " " << I.dur << " " << I.baseV << " " << I.N << " ";
    for (int i = 0; i < I.N; i++) {
        os << I.st[i] << " ";
        os << I.V[i] << " ";
    }
    os << I.fit;
    return os;
}

istream &operator>>(istream &is, inputSpec &I) {
    double tmp;
    I.st.clear();
    I.V.clear();
    is >> I.t >> I.ot >> I.dur >> I.baseV >> I.N;
    for ( int i = 0; i < I.N; i++ ) {
        is >> tmp;
        I.st.push_back(tmp);
        is >> tmp;
        I.V.push_back(tmp);
    }
    is >> I.fit;
    return is;
}
