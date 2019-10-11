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


#ifndef CONDITIONVARIABLE_H
#define CONDITIONVARIABLE_H

#include <memory>

namespace RTMaybe {
    class ConditionVariable;
}

class RTMaybe::ConditionVariable
{
public:
    ConditionVariable();
    ~ConditionVariable();

    ConditionVariable(const ConditionVariable&) = delete;
    ConditionVariable &operator=(const ConditionVariable&) = delete;

    void wait();
    bool wait_if(); /// @return true if the semaphore was taken successfully; false if it is occupied
    bool wait_timed(long nanos); /// Same here. Not implemented in non-RT build.
    void signal();
    void broadcast();

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

#endif // CONDITIONVARIABLE_H
