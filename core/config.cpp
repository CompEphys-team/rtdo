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

conf::Config::Config()
{}


conf::VCConfig::VCConfig() {
    Init();
}
void conf::VCConfig::Init() {
    popsize = 1000;
    wavefile = "";
    in = 0;
    out = 0;
}


conf::IOConfig::IOConfig() {
    Init();
}
void conf::IOConfig::Init() {
    dt = 0.25;
    ai_supersampling = 1;
    channels.clear();
}


conf::OutputConfig::OutputConfig() {
    Init();
}
void conf::OutputConfig::Init() {
    dir = "/tmp";
}


conf::ModelConfig::ModelConfig() {
    Init();
}
void conf::ModelConfig::Init() {
    deffile = "";
    obj = NULL;
}
void conf::ModelConfig::load() {
    if ( obj && !deffile.compare(loadedObjFile) )
        return;
    else if ( obj )
        delete obj;
    obj = new XMLModel(deffile);
    loadedObjFile = deffile;
}


conf::WaveGenConfig::WaveGenConfig() {
    Init();
}
void conf::WaveGenConfig::Init() {
    popsize = 1000;
    ngen = 400;
}


bool conf::Config::save(string filename)
{
    TiXmlDocument doc;
    doc.LinkEndChild(new TiXmlDeclaration("1.0", "", ""));
    TiXmlElement *root = new TiXmlElement("rtdo");
    doc.LinkEndChild(root);

    TiXmlElement *section;
    TiXmlElement *el;

    { // VC
        section = new TiXmlElement("voltageclamp");
        if ( vc.in ) {
            vector<daq_channel *>::iterator it = find(io.channels.begin(), io.channels.end(), vc.in);
            if ( it != io.channels.end() )
                section->SetAttribute("in", (int)(it - io.channels.begin()));
        }
        if ( vc.out ) {
            vector<daq_channel *>::iterator it = find(io.channels.begin(), io.channels.end(), vc.out);
            if ( it != io.channels.end() )
                section->SetAttribute("out", (int)(it - io.channels.begin()));
        }
        section->SetAttribute("popsize", vc.popsize);
        root->LinkEndChild(section);

        el = new TiXmlElement("wavefile");
        el->LinkEndChild(new TiXmlText(vc.wavefile));
        section->LinkEndChild(el);
    }

    { // I/O
        section = new TiXmlElement("io");
        section->SetDoubleAttribute("dt", io.dt);
        section->SetAttribute("ai_supersampling", io.ai_supersampling);
        root->LinkEndChild(section);

        vector<daq_channel *>::iterator it;
        for ( it = io.channels.begin(); it != io.channels.end(); ++it ) {
            const daq_channel *c = *it;
            el = new TiXmlElement("channel");
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
                vector<daq_channel *>::iterator src = find(io.channels.begin(), io.channels.end(), c->read_offset_src);
                if ( src != io.channels.end() ) {
                    sub = new TiXmlElement("offset_source");
                    sub->SetAttribute("channel", (int)(src - io.channels.begin()));
                    sub->SetAttribute("read_later", (int)c->read_offset_later);
                    el->LinkEndChild(sub);
                }
            }
        }
    }

    { // Model
        section = new TiXmlElement("model");
        root->LinkEndChild(section);

        el = new TiXmlElement("deffile");
        el->LinkEndChild(new TiXmlText(model.deffile));
        section->LinkEndChild(el);
    }

    { // Output
        section = new TiXmlElement("output");
        root->LinkEndChild(section);

        el = new TiXmlElement("dir");
        el->LinkEndChild(new TiXmlText(output.dir));
        section->LinkEndChild(el);
    }

    { // WaveGen
        section = new TiXmlElement("wavegen");
        root->LinkEndChild(section);
        section->SetAttribute("popsize", wg.popsize);
        section->SetAttribute("generations", wg.ngen);
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
    io.Init();
    if ( (section = hRoot.FirstChildElement("io").Element()) ) {
        section->QueryDoubleAttribute("dt", &io.dt);
        section->QueryIntAttribute("ai_supersampling", &io.ai_supersampling);
        vector<int> src_idx;
        for ( el = section->FirstChildElement("channel"); el; el = el->NextSiblingElement() ) {
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

            if ( daq_setup_channel(c) )
                continue;
            if ( rtdo_add_channel(c, 10000) )
                continue;

            io.channels.push_back(c);
        }

        // Resolve read_offset_source index into pointers
        vector<daq_channel *>::iterator cit = io.channels.begin();
        vector<int>::iterator pit = src_idx.begin();
        for ( ; pit != src_idx.end(); ++pit, ++cit ) {
            while ( !(*cit)->read_offset_src )
                ++cit;
            if ( *pit > 0 && *pit < (int)io.channels.size() )
                (*cit)->read_offset_src = io.channels[*pit];
            else
                (*cit)->read_offset_src = 0;
        }
    }

    // VC
    vc.Init();
    if ( (section = hRoot.FirstChildElement("voltageclamp").Element()) ) {
        int p = -1;
        section->QueryIntAttribute("in", &p);
        if ( p >= 0 && p < (int)io.channels.size() )
            vc.in = io.channels[p];
        p = -1;
        section->QueryIntAttribute("out", &p);
        if ( p >= 0 && p < (int)io.channels.size() )
            vc.out = io.channels[p];
        section->QueryIntAttribute("popsize", &vc.popsize);
        if ( (el = section->FirstChildElement("wavefile")) )
            vc.wavefile = el->GetText();
    }

    // Model
    model.Init();
    if ( (section = hRoot.FirstChildElement("model").Element()) ) {
        if ( (el = section->FirstChildElement("deffile")) ) {
            model.deffile = el->GetText();
        }
    }

    // Output
    output.Init();
    if ( (section = hRoot.FirstChildElement("output").Element()) ) {
        if ( (el = section->FirstChildElement("dir")) ) {
            output.dir = el->GetText();
        }
    }

    // WaveGen
    wg.Init();
    if ( (section = hRoot.FirstChildElement("wavegen").Element()) ) {
        section->QueryIntAttribute("popsize", &wg.popsize);
        section->QueryIntAttribute("generations", &wg.ngen);
    }

    return true;
}
