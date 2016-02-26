/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-21

--------------------------------------------------------------------------*/
#ifndef SHARED_H
#define SHARED_H

#include <iostream>
#include <queue>
#include <vector>
#include <list>
#include "realtimeconditionvariable.h"
#include "realtimethread.h"

struct inputSpec {
    double t;
    double ot;
    double dur;
    double baseV;
    int N;
    std::vector<double> st;
    std::vector<double> V;
    double fit;
};
std::ostream &operator<<(std::ostream &os, inputSpec &I);
std::istream &operator>>(std::istream &is, inputSpec &I);

struct errTupel
{
    unsigned int id;
    double err;
};

enum stageEnum {
    stDetuneAdjust,
    stNoveltySearch,
    stWaveformOptimise,

    stObservationWindow__start,
    stObservationWindowSeparation,
    stObservationWindowBest,
    stObservationWindowExceed,
    stObservationWindow__end
};

#endif // SHARED_H
