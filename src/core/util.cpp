#include "util.h"
#include <sstream>

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

std::istream &operator>>(std::istream &is, QuotedString &str)
{
    char c = '\0';
    enum {Preamble, Quoted, Unquoted, Done} state = Preamble;
    std::stringstream ss;
    while ( state != Done && (c = is.get()) != EOF ) {
        switch ( state ) {
        case Preamble:
            if ( !QChar(c).isSpace() ) {
                if ( c == '"' ) {
                    state = Quoted;
                } else {
                    state = Unquoted;
                    ss << c;
                }
            }
            break;
        case Quoted:
            if ( c == '"' )
                state = Done;
            else
                ss << c;
            break;
        case Unquoted:
            if ( QChar(c).isSpace() )
                state = Done;
            else
                ss << c;
            break;
        case Done:
            break;
        }
    }
    str = QString::fromStdString(ss.str());
    return is;
}

std::ostream &operator<<(std::ostream &os, const QuotedString &str)
{
    os << '"' << str.toStdString() << '"';
    return os;
}
