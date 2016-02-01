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
    out(0)
{}

void conf::VCConfig::fromXML(TiXmlElement *section, const conf::IOConfig &io) {
    section->QueryIntAttribute("in", &in);
    section->QueryIntAttribute("out", &out);
    section->QueryIntAttribute("popsize", &popsize);

    TiXmlElement *el;
    if ( (el = section->FirstChildElement("wavefile")) )
        wavefile = el->GetText();
}

void conf::VCConfig::toXML(TiXmlElement *section, const conf::IOConfig &io) const
{
    section->SetAttribute("in", in);
    section->SetAttribute("out", out);
    section->SetAttribute("popsize", popsize);

    TiXmlElement *el = new TiXmlElement("wavefile");
    el->LinkEndChild(new TiXmlText(wavefile));
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
        Channel::Type type;
        int deviceno = 0, id = 0;
        unsigned int channel = 0, range = 0, aref = 0;
#ifdef CONFIG_RT
        type = Channel::AnalogIn;
#else
        type = Channel::Simulator;
#endif
        el->QueryIntAttribute("ID", &id);

        if ( (sub = el->FirstChildElement("type")) ) {
            std::string tmp = sub->GetText();
            if ( !tmp.compare("AnalogIn") )
                type = Channel::AnalogIn;
            else if ( !tmp.compare("AnalogOut") )
                type = Channel::AnalogOut;
            else
                type = Channel::Simulator;
        }

        if ( (sub = el->FirstChildElement("device")) ) {
            sub->QueryIntAttribute("number", &deviceno);
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
            channels.back().setName(sub->GetText());

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
        switch ( c.type() ) {
        case Channel::AnalogIn:  sub->SetAttribute("type", "AnalogIn");  break;
        case Channel::AnalogOut: sub->SetAttribute("type", "AnalogOut"); break;
        case Channel::Simulator: sub->SetAttribute("type", "Simulator"); break;
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
        dir = el->GetText();
    }
}

void conf::OutputConfig::toXML(TiXmlElement *section) const
{
    TiXmlElement *el = new TiXmlElement("dir");
    el->LinkEndChild(new TiXmlText(dir));
    section->LinkEndChild(el);
}


conf::ModelConfig::ModelConfig() :
    obj(NULL)
{}

void conf::ModelConfig::fromXML(TiXmlElement *section)
{
    TiXmlElement *el;
    if ( (el = section->FirstChildElement("deffile")) ) {
        deffile = el->GetText();
    }
}

void conf::ModelConfig::toXML(TiXmlElement *section) const
{
    TiXmlElement *el = new TiXmlElement("deffile");
    el->LinkEndChild(new TiXmlText(deffile));
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
    ngen(400)
{}

void conf::WaveGenConfig::fromXML(TiXmlElement *section)
{
    section->QueryIntAttribute("popsize", &popsize);
    section->QueryIntAttribute("generations", &ngen);
}

void conf::WaveGenConfig::toXML(TiXmlElement *section) const
{
    section->SetAttribute("popsize", popsize);
    section->SetAttribute("generations", ngen);
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
        vc.toXML(section, io);
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

    // I/O
    if ( (section = hRoot.FirstChild("io").Element()) ) {
        io.fromXML(section);
    }

    // VC
    if ( (section = hRoot.FirstChild("voltageclamp").Element()) ) {
        vc.fromXML(section, io);
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
