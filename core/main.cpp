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
#include "globals.h"
#include "softrtdaq.h"
#include "rt.h"
#include "config.h"

conf::Config *config;

int main(int argc, char *argv[])
{
    // Set up RT
    int ret=0;
    if ( (ret = rtdo_init()) )
        return ret;
//    rtdo_set_channel_active(daqchan_vout.handle, 1);
//    rtdo_set_channel_active(daqchan_cin.handle, 1);

    config = new conf::Config;

    // Set up GUI
    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    // Run GUI
    ret = a.exec();

    // Cleanup
    rtdo_exit();
    daq_exit();

    return ret;
}
