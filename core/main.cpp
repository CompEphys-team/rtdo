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

conf::Config *config;

int main(int argc, char *argv[])
{
    // Set up RT
    RealtimeEnvironment::RealtimeEnvironment::env();

    config = new conf::Config;

    // Set up GUI
    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    // Run GUI
    return a.exec();
}
