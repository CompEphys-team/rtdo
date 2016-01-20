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
#include "softrtdaq.h"
#include "rt.h"


conf::VCConfig::VCConfig() :
    popsize(1000),
    in(0),
    out(0)
{}

void conf::VCConfig::fromXML(TiXmlElement *section, const conf::IOConfig &io) {
    TiXmlElement *el;
    int p = -1;
    section->QueryIntAttribute("in", &p);
    if ( p >= 0 && p < (int)io.channels.size() )
        in = io.channels[p];

    p = -1;
    section->QueryIntAttribute("out", &p);
    if ( p >= 0 && p < (int)io.channels.size() )
        out = io.channels[p];

    section->QueryIntAttribute("popsize", &popsize);

    if ( (el = section->FirstChildElement("wavefile")) )
        wavefile = el->GetText();
}

void conf::VCConfig::toXML(TiXmlElement *section, const conf::IOConfig &io) const
{
    if ( in ) {
        vector<daq_channel *>::const_iterator it = find(io.channels.begin(), io.channels.end(), in);
        if ( it != io.channels.end() )
            section->SetAttribute("in", (int)(it - io.channels.begin()));
    }
    if ( out ) {
        vector<daq_channel *>::const_iterator it = find(io.channels.begin(), io.channels.end(), out);
        if ( it != io.channels.end() )
            section->SetAttribute("out", (int)(it - io.channels.begin()));
    }
    section->SetAttribute("popsize", popsize);

    TiXmlElement *el = new TiXmlElement("wavefile");
    el->LinkEndChild(new TiXmlText(wavefile));
    section->LinkEndChild(el);
}


conf::IOConfig::IOConfig() :
    dt(0.25),
    ai_supersampling(1)
{}

conf::IOConfig::~IOConfig()
{
    for ( vector<daq_channel *>::iterator it = channels.begin(); it != channels.end(); ++it ) {
        daq_delete_channel(*it);
        delete *it;
    }
}

void conf::IOConfig::fromXML(TiXmlElement *section)
{
    section->QueryDoubleAttribute("dt", &dt);
    section->QueryIntAttribute("ai_supersampling", &ai_supersampling);
    vector<int> src_idx;
    for ( TiXmlElement *el = section->FirstChildElement("channel"); el; el = el->NextSiblingElement() ) {
        daq_channel *c = new daq_channel;
        daq_create_channel(c);

        TiXmlElement *sub;
        if ( (sub = el->FirstChildElement("name")) )
            daq_set_channel_name(c, sub->GetText());

        if ( (sub = el->FirstChildElement("device")) ) {
            sub->QueryUnsignedAttribute("number", &c->deviceno);
            int type = c->type;
            sub->QueryIntAttribute("type", &type);
            c->type = type == COMEDI_SUBD_AO ? COMEDI_SUBD_AO : COMEDI_SUBD_AI;
        }

        if ( (sub = el->FirstChildElement("link")) ) {
            sub->QueryValueAttribute("channel", &c->channel);
            sub->QueryValueAttribute("range", &c->range);
            sub->QueryValueAttribute("aref", &c->aref);
        }

        if ( (sub = el->FirstChildElement("amp")) ) {
            sub->QueryDoubleAttribute("offset", &c->offset);
            sub->QueryDoubleAttribute("conversion_factor", &c->gain);
        }

        if ( (sub = el->FirstChildElement("offset_source")) ) {
            int src = -1;
            sub->QueryIntAttribute("channel", &src);
            src_idx.push_back(src);
            c->read_offset_src = (daq_channel *)1;
            bool later = false;
            sub->QueryBoolAttribute("read_later", &later);
            c->read_offset_later = (char)later;
        }

        if ( daq_setup_channel(c) || rtdo_add_channel(c, 10000) ) {
            daq_delete_channel(c);
            delete c;
            continue;
        }

        channels.push_back(c);
    }

    // Resolve read_offset_source index into pointers
    vector<daq_channel *>::iterator cit = channels.begin();
    vector<int>::iterator pit = src_idx.begin();
    for ( ; pit != src_idx.end(); ++pit, ++cit ) {
        while ( !(*cit)->read_offset_src )
            ++cit;
        if ( *pit > 0 && *pit < (int)channels.size() )
            (*cit)->read_offset_src = channels[*pit];
        else
            (*cit)->read_offset_src = 0;
    }
}

void conf::IOConfig::toXML(TiXmlElement *section) const
{
    section->SetDoubleAttribute("dt", dt);
    section->SetAttribute("ai_supersampling", ai_supersampling);

    for ( vector<daq_channel *>::const_iterator it = channels.begin(); it != channels.end(); ++it ) {
        const daq_channel *c = *it;
        TiXmlElement *el = new TiXmlElement("channel");
        section->LinkEndChild(el);

        TiXmlElement *sub = new TiXmlElement("name");
        sub->LinkEndChild(new TiXmlText(c->name));
        el->LinkEndChild(sub);

        sub = new TiXmlElement("device");
        sub->SetAttribute("number", c->deviceno);
        sub->SetAttribute("type", c->type);
        el->LinkEndChild(sub);

        sub = new TiXmlElement("link");
        sub->SetAttribute("channel", c->channel);
        sub->SetAttribute("range", c->range);
        sub->SetAttribute("aref", c->aref);
        el->LinkEndChild(sub);

        sub = new TiXmlElement("amp");
        sub->SetDoubleAttribute("offset", c->offset);
        sub->SetDoubleAttribute("conversion_factor", c->gain);
        el->LinkEndChild(sub);

        if ( c->read_offset_src ) {
            vector<daq_channel *>::const_iterator src = find(channels.begin(), channels.end(), c->read_offset_src);
            if ( src != channels.end() ) {
                sub = new TiXmlElement("offset_source");
                sub->SetAttribute("channel", (int)(src - channels.begin()));
                sub->SetAttribute("read_later", (int)c->read_offset_later);
                el->LinkEndChild(sub);
            }
        }
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
