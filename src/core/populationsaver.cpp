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


#include "populationsaver.h"

void PopSaver::open(const QFile &basefile)
{
    close();
    pop.open(basefile.fileName().toStdString() + ".pops", std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
    err.open(basefile.fileName().toStdString() + ".errs", std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
}

void PopSaver::close()
{
    if ( pop.is_open() )    pop.close();
    if ( err.is_open() )    err.close();
    pop.clear();
    err.clear();
}

void PopSaver::savePop(UniversalLibrary &lib)
{
    for ( const AdjustableParam &p : lib.adjustableParams )
        pop.write(reinterpret_cast<char*>(p.v), lib.NMODELS * sizeof(*p.v));
}

void PopSaver::saveErr(UniversalLibrary &lib)
{
    err.write(reinterpret_cast<char*>(lib.summary), lib.NMODELS * sizeof(*lib.summary));
}

void PopLoader::open(const QFile &basefile, const UniversalLibrary &lib)
{
    size_t bytes_per_model = 0, bytes_per_err = sizeof(*lib.summary);
    for ( const AdjustableParam &p : lib.adjustableParams )
        bytes_per_model += sizeof(*p.v);

    close();

    QFile combinedf(basefile.fileName() + ".pop");
    QFile popf(basefile.fileName() + ".pops");
    QFile errf(basefile.fileName() + ".errs");

    if ( popf.exists() && errf.exists() ) {
        pop.open(popf.fileName().toStdString(), std::ios_base::in | std::ios_base::binary | std::ios_base::ate);
        err.open(errf.fileName().toStdString(), std::ios_base::in | std::ios_base::binary | std::ios_base::ate);
        combined = false;
        pop_bytes = bytes_per_model * lib.NMODELS;
        err_bytes = bytes_per_err * lib.NMODELS;
        if ( pop.good() && err.good() )
            nEpochs = std::min(pop.tellg() / pop_bytes, err.tellg() / err_bytes);
    } else if ( combinedf.exists() ) {
        pop.open(popf.fileName().toStdString(), std::ios_base::in | std::ios_base::binary | std::ios_base::ate);
        combined = true;
        pop_bytes = (bytes_per_model + bytes_per_err) * lib.NMODELS;
        if ( pop.good() )
            nEpochs = pop.tellg() / pop_bytes;
    }
}

void PopLoader::close()
{
    if ( pop.is_open() )    pop.close();
    if ( err.is_open() )    err.close();
    pop.clear();
    err.clear();
    nEpochs = 0;
}

bool PopLoader::load(int epoch, UniversalLibrary &lib, int param)
{
    if ( epoch >= nEpochs )
        return false;

    pop.seekg(pop_bytes * epoch);

    if ( param < 0 || param >= int(lib.adjustableParams.size()) ) {
        for ( AdjustableParam &p : lib.adjustableParams )
            pop.read(reinterpret_cast<char*>(p.v), lib.NMODELS * sizeof(*p.v));

        if ( !combined ) {
            err.seekg(err_bytes * epoch);
            err.read(reinterpret_cast<char*>(lib.summary), lib.NMODELS * sizeof(*lib.summary));
            return pop.good() && err.good();
        } else {
            pop.read(reinterpret_cast<char*>(lib.summary), lib.NMODELS * sizeof(*lib.summary));
            return pop.good();
        }
    } else {
        for ( int i = 0; i < param; i++ )
            pop.seekg(sizeof(*lib.adjustableParams.at(i).v) * lib.NMODELS, std::ios_base::cur);
        pop.read(reinterpret_cast<char*>(lib.adjustableParams.at(param).v), lib.NMODELS * sizeof(*lib.adjustableParams.at(param).v));
        return pop.good();
    }
}
