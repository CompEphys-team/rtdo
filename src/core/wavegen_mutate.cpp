#include "wavegen.h"
#include <cassert>

Stimulation Wavegen::getRandomStim()
{
    Stimulation I;
    int failedPos, failedAgain = 0;
    I.baseV = p.baseV;
    I.duration = p.duration;
    int n = RNG.uniform(p.minSteps, p.maxSteps);
tryagain:
    failedPos = 0;
    if ( failedAgain++ > 2*p.maxSteps ) {
        failedAgain = 0;
        --n;
    }
    I.steps.clear();
    for ( int i = 0; i < n; i++ ) {
        Stimulation::Step&& step {
            RNG.uniform(p.minStepLength, p.duration - p.minStepLength),
            RNG.uniform(p.minVoltage, p.maxVoltage),
            RNG.pick({true, false})
        };
        bool failed = false;
        auto it = I.steps.begin();
        for ( ; it != I.steps.end(); it++ ) {
            if ( fabs(it->t - step.t) < p.minStepLength ) {
                failed = true;
                break;
            } else if ( it->t > step.t ) {
                failedPos = 0;
                break;
            }
        }
        if ( failed ) {
            if ( ++failedPos > 2*p.maxSteps )
                goto tryagain;
            --i;
            continue;
        } else {
            I.steps.insert(it, std::move(step));
            failedPos = 0;
        }
    }
    return I;
}


Stimulation Wavegen::mutate(const Stimulation &parent, const Stimulation &crossoverParent)
{
    double total = p.muta.lCrossover + p.muta.lLevel + p.muta.lNumber + p.muta.lSwap + p.muta.lTime + p.muta.lType;
    Stimulation I(parent);
    bool didCrossover = false;
    int n = round(RNG.variate<double>(p.muta.n, p.muta.std));
    if ( n < 1 )
        n = 1;
    for ( int i = 0; i < n; i++ ) {
        double selection = RNG.uniform(0.0, total);
        if ( selection < p.muta.lCrossover ) {
            if ( didCrossover ) { // Crossover only once
                --i;
            } else {
                mutateCrossover(I, crossoverParent);
                didCrossover = true;
            }
        } else if ( (selection -= p.muta.lCrossover) < p.muta.lLevel ) {
            mutateVoltage(I);
        } else if ( (selection -= p.muta.lLevel) < p.muta.lNumber ) {
            mutateNumber(I);
        } else if ( (selection -= p.muta.lNumber) < p.muta.lSwap ) {
            mutateSwap(I);
        } else if ( (selection -= p.muta.lSwap) < p.muta.lTime ) {
            mutateTime(I);
        } else /* if ( (selection -= p.muta.lTime) < p.muta.lType ) */ {
            mutateType(I);
        }
    }
    assert([&](){
        bool valid = ((int)I.steps.size() >= p.minSteps) && ((int)I.steps.size() <= p.maxSteps);
        for ( auto it = I.steps.begin(); it != I.steps.end(); it++ ) {
            valid &= (it->t >= p.minStepLength) && (it->t <= p.duration - p.minStepLength);
            if ( it != I.steps.begin() ) {
                valid &= (it->t - (it-1)->t >= p.minStepLength);
            }
            valid &= (it->V >= p.minVoltage) && (it->V <= p.maxVoltage);
        }
        return valid;
    }());
    return I;
}


void Wavegen::mutateCrossover(Stimulation &I, const Stimulation &parent)
{
    std::vector<Stimulation::Step> steps;
    bool coin = RNG.pick({true, false});
    const std::vector<Stimulation::Step> &head = (coin ? I : parent).steps;
    const std::vector<Stimulation::Step> &tail = (coin ? parent : I).steps;
    do { // Repeat on failures
        double mid = RNG.uniform(0.0, p.duration);
        size_t nHead = 0, nTail = 0;
        for ( auto it = head.begin(); it != head.end() && it->t < mid; it++ ) { // Choose all head steps up to mid
            steps.push_back(*it);
            ++nHead;
        }
        for ( auto it = tail.begin(); it != tail.end(); it++ ) { // Choose all tail steps from mid onwards
            if ( it->t > mid ) {
                steps.push_back(*it);
                nTail++;
            }
        }
        if ( (nHead == head.size() && nTail == 0) || (nHead == 0 && nTail == tail.size()) ) // X failed
            continue;
        if ( nHead > 0 && (tail.end()-nTail-1)->t - (head.begin()+nHead-1)->t < p.minStepLength ) { // X step too short
            if ( (int)steps.size() > p.minSteps ) {
                if ( RNG.pick({true, false}) ) {
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
        while ( (int)steps.size() > p.maxSteps ) // Too many steps, delete some
            steps.erase(RNG.choose(steps));
        int count = 0;
        for ( ; (int)steps.size() < p.minSteps && count < 2*p.maxSteps; count++ ) { // Too few steps, insert some
            // Note: count safeguards against irrecoverable failures
            auto extra = head.begin();
            bool useHead = RNG.pick({true, false});
            if ( (useHead && nHead == head.size()) || (!useHead && nTail == tail.size()) )
                continue;
            if ( useHead )
                extra = RNG.choose(head.begin() + nHead, head.end());
            else
                extra = RNG.choose(tail.begin(), tail.end() - nTail);
            for ( auto it = steps.begin(); it != steps.end(); it++ ) {
                if ( fabs(it->t - extra->t) < p.minStepLength )
                    break;
                if ( it->t > extra->t ) {
                    steps.insert(it, *extra);
                    break;
                }
            }
        }
        if ( count == 2*p.maxSteps )
            continue;

        I.steps = steps;
        return;
    } while ( true );
}

void Wavegen::mutateVoltage(Stimulation &I)
{
    Stimulation::Step &subject = RNG.pick(I.steps);
    double newV = p.maxVoltage + 1;
    while ( newV > p.maxVoltage || newV < p.minVoltage )
        newV = RNG.variate<double>(subject.V, p.muta.sdLevel);
    subject.V = newV;
}

void Wavegen::mutateNumber(Stimulation &I)
{
    bool grow;
    if ( (int)I.steps.size() == p.minSteps )
        grow = true;
    else if ( (int)I.steps.size() == p.maxSteps )
        grow = false;
    else
        grow = RNG.pick({true, false});
    if ( grow ) {
        Stimulation::Step ins;
        ins.V = RNG.uniform(p.minVoltage, p.maxVoltage);
        ins.ramp = RNG.pick({true, false});
        bool inserted;
        do {
            ins.t = RNG.uniform(p.minStepLength, p.duration - p.minStepLength);
            bool conflict = false;
            auto it = I.steps.begin();
            for ( ; it != I.steps.end(); it++ ) {
                if ( fabs(it->t - ins.t) < p.minStepLength ) {
                    conflict = true;
                    break;
                }
                if ( it->t > ins.t )
                    break;
            }
            if ( !conflict ) {
                I.steps.insert(it, ins);
                inserted = true;
            }
        } while ( !inserted );
    } else {
        I.steps.erase(RNG.choose(I.steps));
    }
}

void Wavegen::mutateSwap(Stimulation &I)
{
    Stimulation::Step src, dest;
    do {
        src = RNG.pick(I.steps);
        dest = RNG.pick(I.steps);
    } while ( src.t == dest.t );
    using std::swap;
    swap(src.V, dest.V);
    swap(src.ramp, dest.ramp);
}

void Wavegen::mutateTime(Stimulation &I)
{
    bool tooClose = false;
    do {
        auto target = RNG.choose(I.steps);
        double newT;
        do {
            newT = RNG.variate<double>(target->t, p.muta.sdTime);
        } while ( newT < p.minStepLength || newT > p.duration - p.minStepLength );
        auto it = I.steps.begin();
        for ( ; it != I.steps.end(); it++ ) {
            if ( it != target && fabs(it->t - newT) < p.minStepLength ) {
                tooClose = true;
                break;
            }
            if ( it != target && it->t > newT )
                break;
        }
        if ( !tooClose ) {
            if ( it == target+1 ) {
                target->t = newT;
            } else {
                Stimulation::Step tmp = *target;
                tmp.t = newT;
                // Ensure placement is correct; work from tail to head:
                int tOff = target - I.steps.begin(), iOff = it - I.steps.begin();
                if ( tOff < iOff ) {
                    I.steps.insert(it, tmp);
                    I.steps.erase(I.steps.begin() + tOff);
                } else {
                    I.steps.erase(target);
                    I.steps.insert(I.steps.begin() + iOff, tmp);
                }
            }
            return;
        }
    } while ( tooClose );
}

void Wavegen::mutateType(Stimulation &I)
{
    bool &r = RNG.pick(I.steps).ramp;
    r = !r;
}
