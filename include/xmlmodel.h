#ifndef XMLMODEL_H
#define XMLMODEL_H

#include <string>
#include <vector>

class XMLModel
{
public:
    XMLModel();
    XMLModel(std::string filename);
    bool load(std::string filename);

    enum outputType {
        VClamp,
        WaveGen
    };

    struct param {
        std::string name;
        std::string type;
        double initial;
        double min;
        double max;
        double sigma;
        bool multiplicative;
    };

    std::string generateDefinition(enum outputType type, int npop, std::string path);

private:
    std::string name;
    std::string code;
    std::vector<param> vars;
    std::vector<param> adjustableParams;
    std::vector<param> params;
    int precision;
    bool genn_double;
};

#endif // XMLMODEL_H
