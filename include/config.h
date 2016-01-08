/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-08

--------------------------------------------------------------------------*/
#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <vector>
#include <sys/types.h>
#include "types.h"

using namespace std;

namespace conf {
    class Config;
    class VCConfig;
    class IOConfig;
    class OutputConfig;
    class ModelConfig;
}

class conf::VCConfig {
public:
    VCConfig();
    void Init();

    string wavefile;
    string sigfile;
    int popsize;
};

class conf::IOConfig {
public:
    IOConfig();
    void Init();

    vector<daq_channel *> channels;
    double dt;
    int ai_supersampling;
};

class conf::OutputConfig {
public:
    OutputConfig();
    void Init();

    string dir;
};

class conf::ModelConfig {
public:
    ModelConfig();
    void Init();

    string deffile;
};

class conf::Config
{
public:
    Config();

    bool load(string filename);
    bool save(string filename);

    conf::VCConfig vc;
    conf::IOConfig io;
    conf::OutputConfig output;
    conf::ModelConfig model;
};

extern conf::Config config;

#endif // CONFIG_H
