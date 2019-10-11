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


#ifndef PROFILESTATS_H
#define PROFILESTATS_H

#include <QDataStream>
class Session;
class ErrorProfile;

class ProfileStats
{
public:
    ProfileStats(Session &session);

    void process(ErrorProfile *parent, int targetParam);

    struct Single {
        double minValue; //!< Parameter value with the least error
        double minError; //!< Least error
        double deviation; //!< |target param initial value - minValue|
        double localMinima; //!< Number of local minima (to indicate smoothness) [integer value]
        double slope; //!< average dErr/dValue (from separate linear fits on both sides of the global minimum)
        double slopeFit; //!< average sum of squared errors of the two fits
    };

    struct Statistic {
        double mean, median, sd;
    };

    struct Cluster {
        Statistic minValue;
        Statistic minError;
        Statistic deviation;
        Statistic localMinima;
        Statistic slope;
        Statistic slopeFit;

        double index;
        int rank;

        std::vector<Single> single;
    };

    std::vector<Cluster> stats;

    Session &session;

    /// Save/Load
    friend QDataStream &operator<<(QDataStream &os, const ProfileStats &);
    friend QDataStream &operator>>(QDataStream &is, ProfileStats &);
};

#endif // PROFILESTATS_H
