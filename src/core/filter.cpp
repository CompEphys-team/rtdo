#include "filter.h"

// See Gorry 1990, doi:10.1021/ac00205a007
// Since I'm not working with derivatives, I've set s=0 and simplified accordingly in gramPoly.
// Otherwise, this is a direct translation of the PASCAL code in the cited paper.
namespace Gorry
{
double gramPoly(int i, int m, int k)
{
    if ( k > 0 ) {
        return (4*k-2.)/(k*(2*m-k+1)) * i*gramPoly(i,m,k-1) - (k-1)*(2.*m+k)/(k*(2*m-k+1))*gramPoly(i,m,k-2);
    } else {
        return 1;
    }
}

double genFact(int a, int b)
{
    double gf = 1;
    for ( int j = a; j > a-b; j-- )
        gf *= j;
    return gf;
}

double weight(int i, int t, int m, int n)
// "Calculates the weight of the i:th data point for the t:th Least-Square point, over 2m+1 points, order n."
// t=0 returns the Savitzky-Golay weights, while -m<=t<0 and 0<t<=m return weights for the initial and final points, respectively.
{
    double sum = 0;
    for ( int k = 0; k <= n; k++ )
        sum += (2*k+1) * genFact(2*m,k)/genFact(2*m+k+1,k+1) * gramPoly(i,m,k)*gramPoly(t,m,k);
    return sum;
}
}

int cleanWidth(FilterMethod method, int width)
{
    width = int(width/2)*2 + 1;
    switch ( method ) {
    case FilterMethod::MovingAverage:
        return std::max(width, 1);
    case FilterMethod::SavitzkyGolay23:
    case FilterMethod::SavitzkyGolayEdge3:
        return std::max(width, 5);
    case FilterMethod::SavitzkyGolay45:
    case FilterMethod::SavitzkyGolayEdge5:
        return std::max(width, 7);
    }
    return 1;
}

Filter::Filter(FilterMethod method, int w) :
    method(method),
    width(cleanWidth(method, w)),
    kernel(cleanWidth(method, w))
{
    double norm;
    int edgeOrder = 3;
    int m = width/2;
    switch ( method ) {
    case FilterMethod::MovingAverage:
    {
        norm = 1.0/width;
        for ( double &v : kernel )
            v = norm;
        break;
    }
    case FilterMethod::SavitzkyGolay23:
    {
        // See Madden 1978, doi:10.1021/ac50031a048
        // p_s = 3(3m^2 + 3m - 1 - 5s^2) / ((2m+3)(2m+1)(2m-1))
        //     = (3(3m^2 + 3m - 1) - 15s^2) / ((2m+3)(2m+1)(2m-1))
        // Note the use of double precision throughout to prevent integer overflows for large m
        double numerator = 3*(3.*m*m + 3*m - 1);
        norm = 1.0 / ((2.*m+3)*(2*m+1)*(2*m-1));
        for ( int i = 0, s = -m; i < width; i++, s++ ) // s in [-m, m]
            kernel[i] = (numerator - 15.*s*s) * norm;
        break;
    }
    case FilterMethod::SavitzkyGolay45:
    {
        // p_s = 15/4 * ((15m^4 + 30m^3 - 35m^2 - 50m + 12) - 35s^2(2m^2 + 2m - 3) + 63s^4)
        //            / ((2m+5)(2m+3)(2m+1)(2m-1)(2m-3))
        double numerator = (15.*m*m*m*m + 30*m*m*m - 35*m*m - 50*m + 12);
        double ssquareFactor = 35 * (2.*m*m + 2*m - 3);
        norm = 15. / 4. / ((2.*m+5)*(2*m+3)*(2*m+1)*(2*m-1)*(2*m-3));
        for ( int i = 0, s = -m; i < width; i++, s++ ) { // s in [-m, m]
            kernel[i] = (numerator - ssquareFactor*s*s + 63.*s*s*s*s) * norm;
        }
        break;
    }
    case FilterMethod::SavitzkyGolayEdge5:
        edgeOrder = 5;
    case FilterMethod::SavitzkyGolayEdge3:
        edgeKernel.resize(m, std::vector<double>(width));
        for ( int t = 1; t <= m; t++ )
            for ( int i = -m; i <= m; i++ )
                edgeKernel[t-1][i+m] = Gorry::weight(i, t, m, edgeOrder);
        for ( int i = -m; i <= m; i++ )
            kernel[i+m] = Gorry::weight(i, 0, m, edgeOrder);
        break;
    }
}

QVector<double> Filter::filter(const QVector<double> &values)
{
    QVector<double> ret(values.size(), 0);
    if ( method == FilterMethod::SavitzkyGolayEdge3 || method == FilterMethod::SavitzkyGolayEdge5 ) {
        for ( int i = 0; i < values.size(); i++ ) {
            if ( i < int(width/2) ) {
                int e = int(width/2) - 1 - i;
                for ( int k = 0; k < width; k++ )
                    ret[i] += edgeKernel[e][width-1-k] * values[k];

            } else if ( i >= values.size() - int(width/2) ) {
                int e = i - (values.size() - int(width/2));
                for ( int k = 0; k < width; k++ )
                    ret[i] += edgeKernel[e][k] * values[values.size()-width+k];

            } else {
                for ( int k = 0; k < width; k++ )
                    ret[i] += kernel[k] * values[i-int(width/2)+k];
            }
        }
    } else {
        for ( int i = 0; i < values.size(); i++ ) {
            for ( int k = 0; k < width; k++ ) {
                int j = i - int(width/2) + k;
                if ( j < 0 )
                    j = -j;
                else if ( j >= values.size() )
                    j = 2*values.size() - 2 - j;
                ret[i] += kernel[k] * values[j];
            }
        }
    }
    return ret;
}
