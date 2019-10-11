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


#include "thread.h"
#include <thread>

namespace RTMaybe
{

class Thread::Impl
{
public:
    Impl(void *(*fn)(void *), void *arg) :
        t(std::thread(fn, arg))
    {}

    std::thread t;
};

Thread::Thread(void *(*fn)(void *), void *arg, ThreadData) :
    pImpl(new Impl(fn, arg))
{}

Thread::~Thread()
{
    if ( joinable() )
        join();
}

void *Thread::join()
{
    pImpl->t.join();
    return 0;
}

bool Thread::joinable() const
{
    return pImpl->t.joinable();
}

void Thread::initEnv(ThreadData) {}

}
