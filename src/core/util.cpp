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
