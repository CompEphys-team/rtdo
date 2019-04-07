#include "wavegen.h"
#include "session.h"
#include <cassert>

inline bool validStim(iStimulation I, const iStimData &istimd, const StimulationData &stimd)
{
    bool valid = ((int)I.size() >= stimd.minSteps) && ((int)I.size() <= stimd.maxSteps);
    for ( auto it = I.begin(); it != I.end(); it++ ) {
        valid &= (it->t >= istimd.iMinStep) && (it->t <= istimd.iDuration - istimd.iMinStep);
        if ( it != I.begin() ) {
            valid &= (it->t - (it-1)->t >= istimd.iMinStep);
        }
        valid &= (it->V >= stimd.minVoltage) && (it->V <= stimd.maxVoltage);
    }
    return valid;
}

void Wavegen::construct_next_generation(std::vector<iStimulation> &stims)
{
    const int nStims = stims.size();
    std::vector<std::list<MAPElite>::const_iterator> parents(2 * nStims);

    // Sample the archive space with a bunch of random indices
    std::vector<size_t> idx(2 * nStims);
    session.RNG.generate(idx, size_t(0), current.elites.size() - 1);
    std::sort(idx.begin(), idx.end());

    // Insert the sampling into parents in a single run through
    auto archIter = current.elites.begin();
    size_t pos = 0;
    for ( int i = 0; i < 2*nStims; i++ ) {
        std::advance(archIter, idx[i] - pos);
        pos = idx[i];
        parents[i] = archIter;
    }

    // Shuffle the parents, but ensure that xover isn't incestuous
    session.RNG.shuffle(parents);
    for ( int i = 0; i < nStims; i++ ) {
        if ( parents[2*i] == parents[2*i + 1] ) {
            while ( true ) {
                int otherPair = session.RNG.uniform(0, nStims-2);
                otherPair += (otherPair >= i);
                if ( parents[2*otherPair] != parents[2*i] && parents[2*otherPair+1] != parents[2*i] ) {
                    // Both of the other pair are different from the incestuous couple => swap one of them over
                    parents[2*i+1] = parents[2*otherPair];
                    parents[2*otherPair] = parents[2*i];
                    break;
                }
            }
        }
    }

    // Mutate
    for ( int i = 0; i < nStims; i++ ) {
        stims[i] = mutate(*parents[2*i]->wave, *parents[2*i + 1]->wave);
    }
}

iStimulation Wavegen::getRandomStim(const StimulationData &stimd, const iStimData &istimd) const
{
    iStimulation I = iStimulation();
    int failedPos, failedAgain = 0;
    I.baseV = stimd.baseV;
    I.duration = istimd.iDuration;
    int n = session.RNG.uniform(stimd.minSteps, stimd.maxSteps);
tryagain:
    failedPos = 0;
    if ( failedAgain++ > 2*stimd.maxSteps ) {
        failedAgain = 0;
        --n;
    }
    I.clear();
    for ( int i = 0; i < n; i++ ) {
        iStimulation::Step&& step {
            session.RNG.uniform(istimd.iMinStep, istimd.iDuration - istimd.iMinStep),
            session.RNG.uniform(stimd.minVoltage, stimd.maxVoltage),
            session.RNG.pick({true, false})
        };
        bool failed = false;
        auto it = I.begin();
        for ( ; it != I.end(); it++ ) {
            if ( abs(it->t - step.t) < istimd.iMinStep ) {
                failed = true;
                break;
            } else if ( it->t > step.t ) {
                failedPos = 0;
                break;
            }
        }
        if ( failed ) {
            if ( ++failedPos > 2*stimd.maxSteps )
                goto tryagain;
            --i;
            continue;
        } else {
            I.insert(it, std::move(step));
            failedPos = 0;
        }
    }

    if ( stimd.endWithRamp && I.size() < Stimulation::maxSteps )
        I.insert(I.end()-1, iStimulation::Step{istimd.iDuration, session.RNG.uniform(stimd.minVoltage, stimd.maxVoltage), true});

    assert(validStim(I, istimd, stimd));
    return I;
}

iStimulation Wavegen::mutate(const iStimulation &parent, const iStimulation &xoverParent)
{
    double total = stimd.muta.lCrossover + stimd.muta.lLevel + stimd.muta.lNumber + stimd.muta.lSwap + stimd.muta.lTime + stimd.muta.lType;
    iStimulation I(parent);
    bool didCrossover = false;
    int n = round(session.RNG.variate<double>(stimd.muta.n, stimd.muta.std));
    if ( n < 1 )
        n = 1;
    for ( int i = 0; i < n; i++ ) {
        double selection = session.RNG.uniform(0.0, total);
        if ( selection < stimd.muta.lCrossover ) {
            if ( didCrossover ) { // Crossover only once
                --i;
            } else {
                mutateCrossover(I, xoverParent);
                didCrossover = true;
            }
        } else if ( (selection -= stimd.muta.lCrossover) < stimd.muta.lLevel ) {
            mutateVoltage(I);
        } else if ( (selection -= stimd.muta.lLevel) < stimd.muta.lNumber ) {
            mutateNumber(I);
        } else if ( (selection -= stimd.muta.lNumber) < stimd.muta.lSwap ) {
            mutateSwap(I);
        } else if ( (selection -= stimd.muta.lSwap) < stimd.muta.lTime ) {
            mutateTime(I);
        } else /* if ( (selection -= p.muta.lTime) < p.muta.lType ) */ {
            mutateType(I);
        }
    }
    assert(validStim(I, istimd, stimd));
    return I;
}


void Wavegen::mutateCrossover(iStimulation &I, const iStimulation &parent)
{
    int failCount = 0;
    do { // Repeat on failures
        std::vector<iStimulation::Step> steps;
        bool coin = session.RNG.pick({true, false});
        const iStimulation &head = coin ? I : parent;
        const iStimulation &tail = coin ? parent : I;
        int mid = session.RNG.uniform(0, istimd.iDuration);
        size_t nHead = 0, nTail = 0;
        for ( auto it = head.begin(); it != head.end() && it->t < mid; it++ ) { // Choose all head steps up to mid
            steps.push_back(*it);
            ++nHead;
        }
        for ( auto it = tail.begin(); it != tail.end(); it++ ) { // Choose all tail steps from mid onwards
            if ( it->t >= mid ) {
                steps.push_back(*it);
                nTail++;
            }
        }
        if ( (nHead == head.size() && nTail == 0) || (nHead == 0 && nTail == tail.size()) ) { // X failed
            continue;
        }
        if ( nHead > 0 && nTail > 0 && (tail.end()-nTail)->t - (head.begin()+nHead-1)->t < istimd.iMinStep ) { // X step too short
            if ( (int)steps.size() > stimd.minSteps ) {
                if ( session.RNG.pick({true, false}) ) {
                    steps.erase(steps.begin() + nHead);
                    nTail--;
                } else {
                    steps.erase(steps.begin() + nHead - 1);
                    nHead--;
                }
            } else {
                continue;
            }
        }
        while ( (int)steps.size() > stimd.maxSteps ) // Too many steps, delete some
            steps.erase(session.RNG.choose(steps));
        int count = 0;
        for ( ; (int)steps.size() < stimd.minSteps && count < 2*stimd.maxSteps; count++ ) { // Too few steps, insert some
            // Note: count safeguards against irrecoverable failures
            auto extra = head.begin();
            bool useHead = session.RNG.pick({true, false});
            if ( nTail == tail.size() )         useHead = true;
            else if ( nHead == head.size() )    useHead = false;
            if ( useHead )
                extra = session.RNG.choose(head.begin() + nHead, head.end());
            else
                extra = session.RNG.choose(tail.begin(), tail.end() - nTail);
            for ( auto it = steps.begin(); it != steps.end(); it++ ) {
                if ( abs(it->t - extra->t) < istimd.iMinStep )
                    break;
                if ( it->t > extra->t ) {
                    steps.insert(it, *extra);
                    break;
                }
            }
        }
        if ( count == 2*stimd.maxSteps )
            continue;

        I.clear();
        for ( iStimulation::Step &s : steps )
            I.insert(I.end(), std::move(s));
        return;
    } while ( ++failCount < 10 );
    assert(validStim(I, istimd, stimd));
}

void Wavegen::mutateVoltage(iStimulation &I)
{
    iStimulation::Step &subject = session.RNG.pick(I);
    double newV = stimd.maxVoltage + 1;
    while ( newV > stimd.maxVoltage || newV < stimd.minVoltage )
        newV = session.RNG.variate<double>(subject.V, stimd.muta.sdLevel);
    subject.V = newV;
    assert(validStim(I, istimd, stimd));
}

void Wavegen::mutateNumber(iStimulation &I)
{
    bool grow = session.RNG.pick({true, false});
    if ( (int)I.size() == stimd.minSteps )      grow = true;
    else if ( (int)I.size() == stimd.maxSteps ) grow = false;
    if ( grow ) {
        do {
            iStimulation::Step&& ins {
                session.RNG.uniform(istimd.iMinStep, istimd.iDuration - istimd.iMinStep),
                session.RNG.uniform(stimd.minVoltage, stimd.maxVoltage),
                session.RNG.pick({true, false})
            };
            bool conflict = false;
            auto it = I.begin();
            for ( ; it != I.end(); it++ ) {
                if ( abs(it->t - ins.t) < istimd.iMinStep ) {
                    conflict = true;
                    break;
                }
                if ( it->t > ins.t )
                    break;
            }
            if ( !conflict ) {
                I.insert(it, std::move(ins));
                break;
            }
        } while ( true );
    } else {
        I.erase(session.RNG.choose(I));
    }
    assert(validStim(I, istimd, stimd));
}

void Wavegen::mutateSwap(iStimulation &I)
{
    if ( I.size() < 2 )
        return;
    iStimulation::Step *src, *dest;
    do {
        src = session.RNG.choose(I);
        dest = session.RNG.choose(I);
    } while ( src == dest );
    using std::swap;
    swap(src->V, dest->V);
    swap(src->ramp, dest->ramp);
    assert(validStim(I, istimd, stimd));
}

void Wavegen::mutateTime(iStimulation &I)
{
    iStimulation::Step *target;
    int newT;
    bool tooClose;
    auto it = I.begin();
    do {
        tooClose = false;
        target = session.RNG.choose(I);
        do {
            newT = lrint(session.RNG.variate<double>(target->t, stimd.muta.sdTime / session.runData().dt));
        } while ( newT < istimd.iMinStep || newT > istimd.iDuration - istimd.iMinStep );
        for ( it = I.begin(); it != I.end(); it++ ) {
            if ( it != target && abs(it->t - newT) < istimd.iMinStep ) {
                tooClose = true;
                break;
            }
            if ( it != target && it->t > newT ) {
                break;
            }
        }
    } while ( tooClose );

    target->t = newT;
    if ( it != target+1 ) {
        iStimulation::Step tmp = *target;
        I.erase(target);
        I.insert(it - (it > target), std::move(tmp));
    }
    assert(validStim(I, istimd, stimd));
}

void Wavegen::mutateType(iStimulation &I)
{
    bool &r = session.RNG.pick(I).ramp;
    r = !r;
    assert(validStim(I, istimd, stimd));
}
