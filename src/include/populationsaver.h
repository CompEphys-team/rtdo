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


#ifndef POPULATIONSAVER_H
#define POPULATIONSAVER_H

#include "universallibrary.h"
#include <QFile>
#include <fstream>

struct PopSaver {
    std::ofstream pop;
    std::ofstream err;
    void open(const QFile &basefile);
    void close();
    void savePop(UniversalLibrary &lib);
    void saveErr(UniversalLibrary &lib);
    inline PopSaver(const QFile &basefile) { open(basefile); }
};

struct PopLoader {
    std::ifstream pop;
    std::ifstream err;
    bool combined;
    int nEpochs;
    size_t pop_bytes, err_bytes;
    void open(const QFile &basefile, const UniversalLibrary &lib);
    void close();
    bool load(int epoch, UniversalLibrary &lib, int param = -1);
    inline PopLoader(const QFile &basefile, const UniversalLibrary &lib) { open(basefile, lib); }
};


#endif // POPULATIONSAVER_H
