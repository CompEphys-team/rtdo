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
    bool load(int epoch, UniversalLibrary &lib);
    inline PopLoader(const QFile &basefile, const UniversalLibrary &lib) { open(basefile, lib); }
};


#endif // POPULATIONSAVER_H
