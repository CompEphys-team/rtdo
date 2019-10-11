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


#include "conditionvariable.h"
#include <condition_variable>

namespace RTMaybe {

class ConditionVariable::Impl
{
public:
    Impl() :
        count(0),
        release(0)
    {}

    std::condition_variable v;
    std::mutex m;
    int count;
    int release;
};

ConditionVariable::ConditionVariable() :
    pImpl(new Impl)
{}

ConditionVariable::~ConditionVariable()
{
    bool retry;
    do {
        broadcast();
        pImpl->m.lock();
        retry = ( pImpl->count < 0 );
        pImpl->m.unlock();
    } while ( retry );
}

void ConditionVariable::wait()
{
    std::unique_lock<std::mutex> lock(pImpl->m);
    --pImpl->count;
    while ( !pImpl->release ) pImpl->v.wait(lock);
    --pImpl->release;
    ++pImpl->count;
}

bool ConditionVariable::wait_if()
{
    std::unique_lock<std::mutex> lock(pImpl->m);
    if ( pImpl->count > 0 ) {
        --pImpl->count;
        return true;
    } else {
        return false;
    }
}

void ConditionVariable::signal()
{
    std::unique_lock<std::mutex> lock(pImpl->m);
    pImpl->release = 1;
    pImpl->v.notify_all();
}

void ConditionVariable::broadcast()
{
    std::unique_lock<std::mutex> lock(pImpl->m);
    pImpl->release = -1 * pImpl->count;
    pImpl->v.notify_all();
}

}
