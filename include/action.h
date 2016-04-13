/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-04-12

--------------------------------------------------------------------------*/
#ifndef ACTION_H
#define ACTION_H

#include <vector>
#include "tinyxml.h"

class Action
{
public:
    enum Type {
        VCFrontload,
        VCCycle,
        VCRun,
        ModelsSaveAll,
        ModelsSaveEval,
        TracesSave,
        TracesDrop
    };

    Action(Type t, int arg = 0);
    Action(TiXmlElement *section);

    static void writeXML(std::string filename, std::vector<Action>::const_iterator first, std::vector<Action>::const_iterator last );
    static std::vector<Action> readXML(std::string filename);

private:
    Type t;
    int arg;

    void toXml(TiXmlElement *section) const;
    std::string type2text() const;
    Type text2type(std::string txt) const;
};

#endif // ACTION_H
