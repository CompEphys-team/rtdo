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
#include "channel.h"
#include "realtimeenvironment.h"
#include "xmlmodel.h"
#include "tinyxml.h"

using namespace std;

namespace conf {
    class Config;
    class VCConfig;
    class IOConfig;
    class OutputConfig;
    class ModelConfig;
    class WaveGenConfig;
    class RTConfig;
}

class conf::VCConfig {
    friend class conf::Config;
public:
    VCConfig();

    string wavefile;
    int popsize;

    int in;
    int out;

    int gain;
    double resistance;

private:
    void fromXML(TiXmlElement *section, const conf::IOConfig &io);
    void toXML(TiXmlElement *section, const IOConfig &io) const;
};

class conf::IOConfig {
    friend class conf::Config;
public:
    IOConfig();

    vector<Channel> channels;
    double dt;
    int ai_supersampling;

private:
    void fromXML(TiXmlElement *section);
    void toXML(TiXmlElement *section) const;
};

class conf::OutputConfig {
    friend class conf::Config;
public:
    OutputConfig();

    string dir;

private:
    void fromXML(TiXmlElement *section);
    void toXML(TiXmlElement *section) const;
};

class conf::ModelConfig {
    friend class conf::Config;
public:
    ModelConfig();

    bool load(bool forceReload=true);

    string deffile;
    XMLModel *obj;

    int cycles;

private:
    void fromXML(TiXmlElement *section);
    void toXML(TiXmlElement *section) const;

    string loadedObjFile;
};

class conf::WaveGenConfig {
    friend class conf::Config;
public:
    WaveGenConfig();

    int popsize;
    int ngen;

    int ns_ngenOptimise;
    double ns_optimiseProportion;
    double ns_noveltyThreshold;

    double tolTime;
    double tolCurrent;
    double tolDelta;

private:
    void fromXML(TiXmlElement *section);
    void toXML(TiXmlElement *section) const;
};

class conf::RTConfig {
    friend class conf::Config;
public:
    RTConfig();

    unsigned long cpus_ai;
    unsigned long cpus_ao;
    unsigned long cpus_module;
    unsigned long cpus_backlog;

    int prio_ai;
    int prio_ao;
    int prio_module;
    int prio_backlog;

    int ssz_ai;
    int ssz_ao;
    int ssz_module;
    int ssz_backlog;

private:
    void fromXML(TiXmlElement *section);
    void toXML(TiXmlElement *section) const;
};

class conf::Config
{
public:
    Config(string filename = string());

    bool save(string filename);

    conf::VCConfig vc;
    conf::IOConfig io;
    conf::OutputConfig output;
    conf::ModelConfig model;
    conf::WaveGenConfig wg;
    conf::RTConfig rt;

private:
    bool load(string filename);
};

extern conf::Config *config;

#endif // CONFIG_H
