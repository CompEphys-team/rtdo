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


#ifndef THREAD_H
#define THREAD_H

#include <sched.h>
#include "types.h"
#include "conditionvariable.h"

namespace RTMaybe {
    class Thread;
}

class RTMaybe::Thread
{
public:
    /*! Note: Use of Thread in non-RT builds is supported, but the ThreadData argument is ignored.
     * @param priority goes from 0 (high) to 99 (low) in RTAI.
     */
    Thread(void *(*fn)(void *), void *arg, ThreadData p = ThreadData());
    ~Thread();

    Thread(const Thread&) = delete;
    Thread &operator=(const Thread&) = delete;

    void *join();
    bool joinable() const;

    static void initEnv(ThreadData p); /// non-RT: noop. RT: Sets up the calling thread as an RT-capable task.

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

#endif // THREAD_H
