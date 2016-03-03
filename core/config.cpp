/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-08

--------------------------------------------------------------------------*/
#include "config.h"
#include "tinyxml.h"
#include <algorithm>

conf::VCConfig::VCConfig() :
    popsize(1000),
    in(0),
    out(0),
    gain(1000),
    resistance(15.0),
    cacheSize(10),
    cacheUseMedian(true)
{}

void conf::VCConfig::fromXML(TiXmlElement *section) {
    section->QueryIntAttribute("in", &in);
    section->QueryIntAttribute("out", &out);
    section->QueryIntAttribute("popsize", &popsize);

    TiXmlElement *el;
    if ( (el = section->FirstChildElement("wavefile")) )
        wavefile = el->GetText() ? el->GetText() : "";

    if ( (el = section->FirstChildElement("clamp")) ) {
        el->QueryIntAttribute("gain", &gain);
        el->QueryDoubleAttribute("resistance", &resistance);
    }

    if ( (el = section->FirstChildElement("cache")) ) {
        el->QueryUnsignedAttribute("size", &cacheSize);
        el->QueryBoolAttribute("use_median", &cacheUseMedian);
    }
}

void conf::VCConfig::toXML(TiXmlElement *section) const
{
    section->SetAttribute("in", in);
    section->SetAttribute("out", out);
    section->SetAttribute("popsize", popsize);

    TiXmlElement *el = new TiXmlElement("wavefile");
    el->LinkEndChild(new TiXmlText(wavefile));
    section->LinkEndChild(el);

    el = new TiXmlElement("clamp");
    el->SetAttribute("gain", gain);
    el->SetDoubleAttribute("resistance", resistance);
    section->LinkEndChild(el);

    el = new TiXmlElement("cache");
    el->SetAttribute("size", cacheSize);
    el->SetAttribute("use_median", cacheUseMedian ? "true" : "false");
    section->LinkEndChild(el);
}


conf::IOConfig::IOConfig() :
    dt(0.25),
    ai_supersampling(1)
{}

void conf::IOConfig::fromXML(TiXmlElement *section)
{
    section->QueryDoubleAttribute("dt", &dt);
    section->QueryIntAttribute("ai_supersampling", &ai_supersampling);

    for ( TiXmlElement *el = section->FirstChildElement("channel"); el; el = el->NextSiblingElement() ) {
        TiXmlElement *sub;
        Channel::Direction type = Channel::AnalogIn;
        int deviceno = 0, id = 0;
        unsigned int channel = 0, range = 0, aref = 0;

        el->QueryIntAttribute("ID", &id);

        if ( (sub = el->FirstChildElement("device")) ) {
            sub->QueryIntAttribute("number", &deviceno);
            std::string tmp(sub->Attribute("type"));
            type = !tmp.compare("AnalogIn") ? Channel::AnalogIn : Channel::AnalogOut;
        }

        if ( (sub = el->FirstChildElement("link")) ) {
            sub->QueryUnsignedAttribute("channel", &channel);
            sub->QueryUnsignedAttribute("range", &range);
            sub->QueryUnsignedAttribute("aref", &aref);
        }

        try {
            if ( id )
                channels.push_back(Channel(id, type, deviceno, channel, range, static_cast<Channel::Aref>(aref)));
            else
                channels.push_back(Channel(type, deviceno, channel, range, static_cast<Channel::Aref>(aref)));
        } catch ( RealtimeException & ) {
            continue;
        }

        if ( (sub = el->FirstChildElement("name")) )
            channels.back().setName(sub->GetText() ? sub->GetText() : "");

        if ( (sub = el->FirstChildElement("amp")) ) {
            double offset=0, gain=1;
            int src = 0;
            sub->QueryDoubleAttribute("offset", &offset);
            sub->QueryDoubleAttribute("conversion_factor", &gain);
            sub->QueryIntAttribute("offset_source", &src);
            channels.back().setOffset(offset);
            channels.back().setConversionFactor(gain);
            channels.back().setOffsetSource(src);
        }
    }
}

void conf::IOConfig::toXML(TiXmlElement *section) const
{
    section->SetDoubleAttribute("dt", dt);
    section->SetAttribute("ai_supersampling", ai_supersampling);

    for ( const Channel &c : channels ) {
        TiXmlElement *el = new TiXmlElement("channel");
        el->SetAttribute("ID", c.ID());
        section->LinkEndChild(el);

        TiXmlElement *sub = new TiXmlElement("name");
        sub->LinkEndChild(new TiXmlText(c.name()));
        el->LinkEndChild(sub);

        sub = new TiXmlElement("device");
        sub->SetAttribute("number", c.device());
        switch ( c.direction() ) {
        case Channel::AnalogIn:  sub->SetAttribute("type", "AnalogIn");  break;
        case Channel::AnalogOut: sub->SetAttribute("type", "AnalogOut"); break;
        }
        el->LinkEndChild(sub);

        sub = new TiXmlElement("link");
        sub->SetAttribute("channel", c.channel());
        sub->SetAttribute("range", c.range());
        sub->SetAttribute("aref", c.aref());
        el->LinkEndChild(sub);

        sub = new TiXmlElement("amp");
        sub->SetDoubleAttribute("offset", c.offset());
        sub->SetDoubleAttribute("conversion_factor", c.conversionFactor());
        sub->SetAttribute("offset_source", c.offsetSource());
        el->LinkEndChild(sub);
    }
}


conf::OutputConfig::OutputConfig() :
    dir("/tmp")
{}

void conf::OutputConfig::fromXML(TiXmlElement *section)
{
    TiXmlElement *el;
    if ( (el = section->FirstChildElement("dir")) ) {
        dir = el->GetText() ? el->GetText() : "";
    }
}

void conf::OutputConfig::toXML(TiXmlElement *section) const
{
    TiXmlElement *el = new TiXmlElement("dir");
    el->LinkEndChild(new TiXmlText(dir));
    section->LinkEndChild(el);
}


conf::ModelConfig::ModelConfig() :
    obj(NULL),
    cycles(50)
{}

void conf::ModelConfig::fromXML(TiXmlElement *section)
{
    TiXmlElement *el;
    if ( (el = section->FirstChildElement("deffile")) ) {
        deffile = el->GetText() ? el->GetText() : "";
    }

    if ( (el = section->FirstChildElement("resolution")) ) {
        el->QueryIntAttribute("cycles", &cycles);
    }
}

void conf::ModelConfig::toXML(TiXmlElement *section) const
{
    TiXmlElement *el = new TiXmlElement("deffile");
    el->LinkEndChild(new TiXmlText(deffile));
    section->LinkEndChild(el);

    el = new TiXmlElement("resolution");
    el->SetAttribute("cycles", cycles);
    section->LinkEndChild(el);
}

bool conf::ModelConfig::load(bool forceReload) {
    if ( !forceReload && obj && !deffile.compare(loadedObjFile) )
        return true;
    else if ( obj )
        delete obj;
    try {
        obj = new XMLModel(deffile);
    } catch (exception &e) {
        return false;
    }
    loadedObjFile = deffile;
    return true;
}


conf::WaveGenConfig::WaveGenConfig() :
    popsize(1000),
    ngen(400),
    ns_ngenOptimise(100),
    ns_optimiseProportion(0.2),
    ns_noveltyThreshold(0.1),
    tolTime(0.5),
    tolCurrent(0.001),
    tolDelta(0.0005)
{}

void conf::WaveGenConfig::fromXML(TiXmlElement *section)
{
    section->QueryIntAttribute("popsize", &popsize);
    section->QueryIntAttribute("generations", &ngen);

    TiXmlElement *el;
    if ( (el = section->FirstChildElement("noveltysearch")) ) {
        el->QueryDoubleAttribute("threshold", &ns_noveltyThreshold);

        if ( (el = el->FirstChildElement("optimisation")) ) {
            el->QueryIntAttribute("generations", &ns_ngenOptimise);
            el->QueryDoubleAttribute("proportion", &ns_optimiseProportion);
        }
    }

    if ( (el = section->FirstChildElement("tolerance")) ) {
        el->QueryDoubleAttribute("time", &tolTime);
        el->QueryDoubleAttribute("current", &tolCurrent);
        el->QueryDoubleAttribute("delta", &tolDelta);
    }
}

void conf::WaveGenConfig::toXML(TiXmlElement *section) const
{
    section->SetAttribute("popsize", popsize);
    section->SetAttribute("generations", ngen);

    TiXmlElement *ns = new TiXmlElement("noveltysearch");
    ns->SetDoubleAttribute("threshold", ns_noveltyThreshold);
    section->LinkEndChild(ns);

    TiXmlElement *el = new TiXmlElement("optimisation");
    el->SetAttribute("generations", ns_ngenOptimise);
    el->SetDoubleAttribute("proportion", ns_optimiseProportion);
    ns->LinkEndChild(el);

    el = new TiXmlElement("tolerance");
    el->SetDoubleAttribute("time", tolTime);
    el->SetDoubleAttribute("current", tolCurrent);
    el->SetDoubleAttribute("delta", tolDelta);
    section->LinkEndChild(el);
}


conf::RTConfig::RTConfig() :
    cpus_ai(0xFFFF),
    cpus_ao(0xFFFF),
    cpus_module(0xFFFF),
    cpus_backlog(0xFFFF),
    prio_ai(10),
    prio_ao(10),
    prio_module(20),
    prio_backlog(80),
    ssz_ai(8*1024),
    ssz_ao(8*1024),
    ssz_module(256*1024),
    ssz_backlog(256*1024)
{}

void conf::RTConfig::fromXML(TiXmlElement *section)
{
    TiXmlElement *el;
    const char *cpus;
    if ( (el = section->FirstChildElement("analog_in")) ) {
        if ( (cpus = el->Attribute("cpus")) )
            cpus_ai = strtoul(cpus, nullptr, 0);
        el->QueryIntAttribute("prio", &prio_ai);
        el->QueryIntAttribute("stacksz", &ssz_ai);
    }
    if ( (el = section->FirstChildElement("analog_out")) ) {
        if ( (cpus = el->Attribute("cpus")) )
            cpus_ao = strtoul(cpus, nullptr, 0);
        el->QueryIntAttribute("prio", &prio_ao);
        el->QueryIntAttribute("stacksz", &ssz_ao);
    }
    if ( (el = section->FirstChildElement("module")) ) {
        if ( (cpus = el->Attribute("cpus")) )
            cpus_module = strtoul(cpus, nullptr, 0);
        el->QueryIntAttribute("prio", &prio_module);
        el->QueryIntAttribute("stacksz", &ssz_module);
    }
    if ( (el = section->FirstChildElement("backlog")) ) {
        if ( (cpus = el->Attribute("cpus")) )
            cpus_backlog = strtoul(cpus, nullptr, 0);
        el->QueryIntAttribute("prio", &prio_backlog);
        el->QueryIntAttribute("stacksz", &ssz_backlog);
    }
}

void conf::RTConfig::toXML(TiXmlElement *section) const
{
    TiXmlElement *el;
    el = new TiXmlElement("analog_in");
    el->SetAttribute("cpus", cpus_ai);
    el->SetAttribute("prio", prio_ai);
    el->SetAttribute("stacksz", ssz_ai);
    section->LinkEndChild(el);

    el = new TiXmlElement("analog_out");
    el->SetAttribute("cpus", cpus_ao);
    el->SetAttribute("prio", prio_ao);
    el->SetAttribute("stacksz", ssz_ao);
    section->LinkEndChild(el);

    el = new TiXmlElement("module");
    el->SetAttribute("cpus", cpus_module);
    el->SetAttribute("prio", prio_module);
    el->SetAttribute("stacksz", ssz_module);
    section->LinkEndChild(el);

    el = new TiXmlElement("backlog");
    el->SetAttribute("cpus", cpus_backlog);
    el->SetAttribute("prio", prio_backlog);
    el->SetAttribute("stacksz", ssz_backlog);
    section->LinkEndChild(el);
}


conf::Config::Config(string filename)
{
    if ( !filename.empty() ) {
        if ( !load(filename) ) {
            throw("Config failed to parse.");
        }
    }
}

bool conf::Config::save(string filename)
{
    TiXmlDocument doc;
    doc.LinkEndChild(new TiXmlDeclaration("1.0", "", ""));
    TiXmlElement *root = new TiXmlElement("rtdo");
    doc.LinkEndChild(root);

    TiXmlElement *section;

    { // VC
        section = new TiXmlElement("voltageclamp");
        root->LinkEndChild(section);
        vc.toXML(section);
    }

    { // I/O
        section = new TiXmlElement("io");
        root->LinkEndChild(section);
        io.toXML(section);
    }

    { // Model
        section = new TiXmlElement("model");
        root->LinkEndChild(section);
        model.toXML(section);
    }

    { // Output
        section = new TiXmlElement("output");
        root->LinkEndChild(section);
        output.toXML(section);
    }

    { // WaveGen
        section = new TiXmlElement("wavegen");
        root->LinkEndChild(section);
        wg.toXML(section);
    }

    { // RT
        section = new TiXmlElement("RTAI");
        root->LinkEndChild(section);
        rt.toXML(section);
    }

    return doc.SaveFile(filename);
}

bool conf::Config::load(string filename)
{
    TiXmlDocument doc;
    doc.LoadFile(filename);

    TiXmlHandle hDoc(&doc);
    TiXmlElement *el;
    TiXmlElement *section;

    el = hDoc.FirstChildElement().Element();
    if ( !el )
        return false;
    TiXmlHandle hRoot(el);

    // RT
    if ( (section = hRoot.FirstChild("RTAI").Element()) ) {
        rt.fromXML(section);
    }

    RealtimeEnvironment::reboot();

    // I/O
    if ( (section = hRoot.FirstChild("io").Element()) ) {
        io.fromXML(section);
    }

    // VC
    if ( (section = hRoot.FirstChild("voltageclamp").Element()) ) {
        vc.fromXML(section);
    }

    // Model
    if ( (section = hRoot.FirstChildElement("model").Element()) ) {
        model.fromXML(section);
    }

    // Output
    if ( (section = hRoot.FirstChildElement("output").Element()) ) {
        output.fromXML(section);
    }

    // WaveGen
    if ( (section = hRoot.FirstChildElement("wavegen").Element()) ) {
        wg.fromXML(section);
    }

    return true;
}
