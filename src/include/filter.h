#ifndef FILTER_H
#define FILTER_H

#include "types.h"
#include "QVector"

class Filter
{
public:
    Filter(FilterMethod method, int width);
    QVector<double> filter(const QVector<double> &values);

    const FilterMethod method;
    const int width;
    std::vector<double> kernel;
    std::vector<std::vector<double>> edgeKernel;
};

#endif // FILTER_H
