/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-02-15

Note: Original author is Thomas Guest, published in the public domain at <http://wordaligned.org/articles/cpp-streambufs>

--------------------------------------------------------------------------*/
#ifndef TEESTREAM_H
#define TEESTREAM_H
#include <iostream>

class teebuf: public std::streambuf
{
public:
    // Construct a streambuf which tees output to both input
    // streambufs.
    teebuf(std::streambuf * sb1, std::streambuf * sb2);
protected:
    virtual int overflow(int c);
    virtual int sync();
private:
    std::streambuf * sb1;
    std::streambuf * sb2;
};


class teestream : public std::ostream
{
public:
    // Construct an ostream which tees output to the supplied
    // ostreams.
    teestream(std::ostream & o1, std::ostream & o2);
private:
    teebuf tbuf;
};


#endif // TEESTREAM_H
