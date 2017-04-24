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
