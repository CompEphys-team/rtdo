#ifndef SUPPORTCODE_CU
#define SUPPORTCODE_CU

#include "kernelhelper.h"

__device__ void processStats(const scalar err,  //!< Target parameter's absolute deviation from base model on this cycle
                             const scalar next, //!< Highest deviation from a parameter other than the target
                             const scalar mean, //!< Mean deviation across all parameters
                             const scalar t,    //!< Time, including substep contribution
                             GeNN_Bridge::WaveStats &s, //!< Stats struct for this group & target param
                             const bool final)  //!< True if this is the very last cycle (= force close bubbles)
{
    scalar abs, rel, meanAbs, meanRel;

    if ( (mean > err && s.currentBud.cycles) || (final && err > mean) ) {
        s.buds++;
        s.currentBud = {0};

        s.currentBud.tEnd = t;
        if ( s.currentBud.cycles       > s.longestBud.cycles )                s.longestBud = s.currentBud;
        if ( s.currentBud.meanAbs      > s.bestMeanAbsBud.meanAbs )           s.bestMeanAbsBud = s.currentBud;
        if ( s.currentBud.meanRel      > s.bestMeanRelBud.meanRel )           s.bestMeanRelBud = s.currentBud;
        if ( s.currentBud.shortfallAbs > s.bestShortfallAbsBud.shortfallAbs ) s.bestShortfallAbsBud = s.currentBud;
        if ( s.currentBud.shortfallRel > s.bestShortfallRelBud.shortfallRel ) s.bestShortfallRelBud = s.currentBud;
    } else if ( err > mean ) {
        abs = err - next;
        rel = abs / next;
        meanAbs = err - mean;
        meanRel = meanAbs / mean;

        s.currentBud.cycles++;
        s.currentBud.meanAbs += abs;
        s.currentBud.meanRel += rel;
        s.currentBud.shortfallAbs += meanAbs;
        s.currentBud.shortfallRel += meanRel;

        s.totalBud.cycles++;
        s.totalBud.meanAbs += abs;
        s.totalBud.meanRel += rel;
        s.totalBud.shortfallAbs += meanAbs;
        s.totalBud.shortfallRel += meanRel;
    }

    if ( (next > err && s.currentBubble.cycles) || (final && err > next) ) {
    //   a bubble has just ended    or    this is the final cycle and a bubble is still open
    // ==> Close the bubble, collect stats:
        s.bubbles++;
        s.currentBubble = {0};

        s.currentBubble.tEnd = t;
        if ( s.currentBubble.cycles  > s.longestBubble.cycles )         s.longestBubble = s.currentBubble;
        if ( s.currentBubble.abs     > s.bestAbsBubble.abs )            s.bestAbsBubble = s.currentBubble;
        if ( s.currentBubble.rel     > s.bestRelBubble.rel )            s.bestRelBubble = s.currentBubble;
        if ( s.currentBubble.meanAbs > s.bestMeanAbsBubble.meanAbs )    s.bestMeanAbsBubble = s.currentBubble;
        if ( s.currentBubble.meanRel > s.bestMeanRelBubble.meanRel )    s.bestMeanRelBubble = s.currentBubble;
    } else if ( err > next ) {
    // Process open bubble:
    // Use abs, rel, meanAbs, meanRel that were calculated in the bud branch above: Bubbles are a subset of Buds!
        s.currentBubble.cycles++;
        s.currentBubble.abs += abs;
        s.currentBubble.rel += rel;
        s.currentBubble.meanAbs += meanAbs;
        s.currentBubble.meanRel += meanRel;

        s.totalBubble.cycles++;
        s.totalBubble.abs += abs;
        s.totalBubble.rel += rel;
        s.totalBubble.meanAbs += meanAbs;
        s.totalBubble.meanRel += meanRel;
    }
}

#endif // SUPPORTCODE_CU