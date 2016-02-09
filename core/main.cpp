/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2015-12-03

--------------------------------------------------------------------------*/
#include "mainwindow.h"
#include <QApplication>
#include "realtimeenvironment.h"
#include "config.h"
#include "optionparser.h"
#include <vector>
#include "runner.h"

conf::Config *config;

struct Arg: public option::Arg
{
    static void printError(const char* msg1, const option::Option& opt, const char* msg2)
    {
      fprintf(stderr, "%s", msg1);
      fwrite(opt.name, opt.namelen, 1, stderr);
      fprintf(stderr, "%s", msg2);
    }

    static option::ArgStatus NonEmpty(const option::Option& option, bool msg)
    {
      if (option.arg != 0 && option.arg[0] != 0)
        return option::ARG_OK;

      if (msg) printError("Option '", option, "' requires a non-empty argument\n");
      return option::ARG_ILLEGAL;
    }

    static option::ArgStatus Task(const option::Option &option, bool msg)
    {
        if ( option.arg != 0 && option.arg[0] != 0 ) {
            if ( std::string("vclamp").compare(option.arg) &&
                 std::string("wavegen").compare(option.arg) ) {
                if ( msg ) fprintf(stderr, "Unkown task '%s'. Known tasks are 'wavegen' and 'vclamp'.\n", option.arg);
                return option::ARG_ILLEGAL;
            }
            return option::ARG_OK;
        }

        if ( msg ) std::cerr << "Task option (e.g. wavegen, vclamp) required.\n";
        return option::ARG_ILLEGAL;
    }
};

enum optIndex { HELP, TASK, BUILD, RUN, CONFIG, CFG_MODEL, CFG_GENERATIONS };
const option::Descriptor usage[] =
{
    { HELP,         0, "h", "help",     Arg::None,      u8"-h, --help\tPrint this help" },
    { TASK,         0, "t", "task",     Arg::Task,      u8"-tX, --task=X\tComplete task X in this run. "
                                                        u8"Currently supported tasks are 'wavegen' and 'vclamp'." },
    { BUILD,        0, "b", "build",    Arg::None,      u8"-b, --build\tRecompile the model from its definition file." },
    { RUN,          0, "r", "run",      Arg::None,      u8"-r, --run\tRun the specified task." },
    { CONFIG,       0, "c", "config",   Arg::NonEmpty,  u8"-cFILE, --config=FILE\tUse the config xml in FILE." },
    { CFG_MODEL,    0, "m", "model",    Arg::Optional,  u8"-mFILE, --model=FILE\tUse the model xml in FILE, "
                                                        u8"instead of the one specified in the config file." },
    { CFG_GENERATIONS, 0, "g", "generations", Arg::NonEmpty, u8"--generations=X\tRun X generations in waveform generation." },
    {0,0,0,0,0,0}
};

int main(int argc, char *argv[])
{
    // Set up RT and config
    RealtimeEnvironment::RealtimeEnvironment::env();
    config = new conf::Config;

    argc-=(argc>0); argv+=(argc>0); // skip program name argv[0] if present

    if ( argc == 0 ) { // Run in GUI mode
        // Set up GUI
        QApplication a(argc, argv);
        MainWindow w;
        w.show();

        // Run GUI
        try {
            return a.exec();
        } catch ( exception &e ) {
            std::cout << "A fatal exception occurred:" << std::endl << e.what() << std::endl;
            return 1;
        }

    } else { // Run in CLI mode
        option::Stats  stats(usage, argc, argv);
        std::vector<option::Option> options(stats.options_max);
        std::vector<option::Option> buffer(stats.buffer_max);
        option::Parser parse(usage, argc, argv, &options[0], &buffer[0]);

        if ( parse.error() )
          return 1;

        if ( options[HELP] ) {
          option::printUsage(std::cout, usage);
          return 0;
        }

        if ( !options[CONFIG] ) {
            std::cout << "No configuration specified. Please supply one with '-c FILE'." << std::endl;
            return 1;
        }
        try {
            config = new conf::Config(options[CONFIG].last()->arg);
        } catch (...) {
            std::cout << "Failed to open or parse config file '" << options[CONFIG].arg << "'." << std::endl;
            return 1;
        }

        if ( options[CFG_MODEL] ) {
            config->model.deffile = options[CFG_MODEL].last()->arg;
        }
        if ( !config->model.load() ) {
            std::cout << "Failed to load or parse the model file '" << config->model.deffile << "'." << std::endl;
            return 1;
        }

        if ( options[CFG_GENERATIONS] )
            config->wg.ngen = atoi(options[CFG_GENERATIONS].last()->arg );

        if ( !options[BUILD] && !options[RUN] ) {
            std::cout << "Nothing to do. Specify -b to build, and/or -r to run." << std::endl;
            return 1;
        }

        if ( !options[TASK] ) {
            std::cout << "Nothing to do. Specify a task with '-t taskname'." << std::endl;
            return 1;
        }

        bool buildFailed = false;
        if ( options[BUILD] ) {
            CompileRunner *builder = 0;
            for ( option::Option *opt = &options[TASK]; opt; opt = opt->next() ) {
                if ( !std::string("wavegen").compare(opt->arg) )       builder = new CompileRunner(XMLModel::WaveGen);
                else if ( !std::string("vclamp").compare(opt->arg) )   builder = new CompileRunner(XMLModel::VClamp);
                else continue;
                std::cout << "Building " << opt->arg << "..." << std::endl;
                if ( !builder->start() ) {
                    std::cout << "Building " << opt->arg << " failed." << std::endl;
                    buildFailed = true;
                }
            }
        }

        if ( !buildFailed && options[RUN] ) {
            Runner *runner = 0;
            for ( option::Option *opt = &options[TASK]; opt; opt = opt->next() ) {
                if ( !std::string("wavegen").compare(opt->arg) )       runner = new Runner(XMLModel::WaveGen);
                else if ( !std::string("vclamp").compare(opt->arg) )   runner = new Runner(XMLModel::VClamp);
                else continue;
                std::cout << "Running " << opt->arg << "..." << std::endl;
                if ( !runner->start() ) {
                    std::cout << "Running " << opt->arg << " failed." << std::endl;
                }
                runner->wait();
            }
        }

        return 0;
    }
}
