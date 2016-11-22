/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-02-15

Note: Original author is Thomas Guest, published in the public domain at <http://wordaligned.org/articles/cpp-streambufs>

--------------------------------------------------------------------------*/
#include "teestream.h"

teestream::teestream(std::ostream & o1, std::ostream & o2)
  : std::ostream(&tbuf)
  , tbuf(o1.rdbuf(), o2.rdbuf())
{}


teebuf::teebuf(std::streambuf * sb1, std::streambuf * sb2)
    : sb1(sb1)
    , sb2(sb2)
{}

int teebuf::overflow(int c)
{
    if (c == EOF)
    {
        return !EOF;
    }
    else
    {
        int const r1 = sb1->sputc(c);
        int const r2 = sb2->sputc(c);
        return r1 == EOF || r2 == EOF ? EOF : c;
    }
}

int teebuf::sync()
{
    int const r1 = sb1->pubsync();
    int const r2 = sb2->pubsync();
    return r1 == 0 && r2 == 0 ? 0 : -1;
}
