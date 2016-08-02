/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-20

--------------------------------------------------------------------------*/
#ifndef XMLMODEL_H
#define XMLMODEL_H

#include <string>
#include <vector>
#include <fstream>

class XMLModel
{
public:
    XMLModel();
    XMLModel(std::string filename);
    bool load(std::string filename);

    enum outputType {
        VClamp,
        WaveGen,
        WaveGenNoveltySearch
    };

    struct param {
        param() {}
        param(std::string n, std::string t = "scalar") : name(n), type(t) {}
        std::string name;
        std::string type;
        double initial;
        double min;
        double max;
        double sigma;
        bool multiplicative;
    };

    struct Current {
        std::string name;
        std::string code;
    };

    std::string generateDefinition(enum outputType type, int npop, std::string path, bool single = false);
    void generateSimulator(enum outputType type, std::string path);

    inline const std::string &name() const { return _name; }
    std::string name(enum outputType type, bool single = false) const;
    inline const std::vector<param> &vars() const { return _vars; }
    inline const std::vector<param> &adjustableParams() const { return _adjustableParams; }
    inline const std::vector<param> &params() const { return _params; }
    inline const std::vector<Current> &currents() const { return _currents; }
    inline bool genn_float() const { return !genn_double; }
    inline double baseV() const { return _baseV; }

private:
    std::string _name;
    std::string code;
    std::vector<param> _vars;
    std::vector<param> _adjustableParams;
    std::vector<param> _params;
    std::vector<Current> _currents;
    int precision;
    bool genn_double;
    double _baseV;

    void gendef_variables(std::ofstream &of, int nExtraVars, bool addCurrents);
    void gendef_pushVar(std::ofstream &of, const param &p);
    void gendef_pushParam(std::ofstream &of, const param &p);
    void gendef_pushGlobalParam(std::ofstream &of, const param &p);

    void gendef_VClamp(std::ofstream &of, int npop, std::string modelname);
    void gendef_wavegen(std::ofstream &of, int npop, std::string modelname);
    void gendef_wavegenNS(std::ofstream &of, int npop, std::string modelname);
};

#endif // XMLMODEL_H
