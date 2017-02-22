#include "util.h"
#include <QChar>
#include <sstream>

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
    str = ss.str();
    return is;
}

std::ostream &operator<<(std::ostream &os, const QuotedString &str)
{
    os << '"' << std::string(str) << '"';
    return os;
}
