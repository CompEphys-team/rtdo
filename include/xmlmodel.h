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

    inline const std::vector<param> &vars() const { return _vars; }
    inline const std::vector<param> &adjustableParams() const { return _adjustableParams; }
    inline const std::vector<param> &params() const { return _params; }

private:
    std::string name;
    std::string code;
    std::vector<param> _vars;
    std::vector<param> _adjustableParams;
    std::vector<param> _params;
    int precision;
    bool genn_double;
};

#endif // XMLMODEL_H
