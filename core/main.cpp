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

daq_channel daqchan_vout;
daq_channel daqchan_cout;
daq_channel daqchan_vin;
daq_channel daqchan_cin;
struct _sim_params sim_params = SIMPARAM_DEFAULT;

int main(int argc, char *argv[])
{
    int ret=0;
    sim_params.outdir = "/home/felix/projects/build/rtdo/output";
    sim_params.sigfile = "/home/felix/projects/rtdo/models/Lymnaea_B1.dat";
    sim_params.vc_wavefile = "/home/felix/projects/rtdo/vclamp/wave2.dat";
    sim_params.modelfile = "/home/felix/projects/rtdo/models/Lymnaea_B1.cc";

    //--------------------------------------------------------------
    // Set up channels
    daq_create_channel(&daqchan_vout);
    daqchan_vout.type = COMEDI_SUBD_AO;
    daqchan_vout.gain = 20.0;
    if ( (ret = daq_setup_channel(&daqchan_vout)) )
        return ret;

    daq_create_channel(&daqchan_cout);
    daqchan_cout.type = COMEDI_SUBD_AO;
    daqchan_cout.channel = 1;
    if ( (ret = daq_setup_channel(&daqchan_cout)) )
        return ret;

    daq_create_channel(&daqchan_vin);
    daqchan_vin.type = COMEDI_SUBD_AI;
    daqchan_vin.aref = AREF_DIFF;
    daqchan_vin.gain = 100.0;
    if ( (ret = daq_setup_channel(&daqchan_vin)) )
        return ret;

    daq_create_channel(&daqchan_cin);
    daqchan_cin.type = COMEDI_SUBD_AI;
    daqchan_cin.aref = AREF_DIFF;
    daqchan_cin.channel = 1;
    daqchan_cin.gain = 10.0;
    if ( (ret = daq_setup_channel(&daqchan_cin)) )
        return ret;

    //--------------------------------------------------------------
    // Set up RT
    if ( (ret = rtdo_init()) )
        return ret;
    if ( (ret = rtdo_add_channel(&daqchan_vout, 10)) )
        return ret;
    if ( (ret = rtdo_add_channel(&daqchan_cout, 10)) )
        return ret;
    if ( (ret = rtdo_add_channel(&daqchan_vin, 10000)) )
        return ret;
    if ( (ret = rtdo_add_channel(&daqchan_cin, 10000)) )
        return ret;
    rtdo_set_channel_active(daqchan_cout.handle, 0);

    //--------------------------------------------------------------
    // Set up GUI
    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    // Run GUI
    ret = a.exec();

    // Cleanup
    rtdo_exit();

    daq_delete_channel(&daqchan_vout);
    daq_delete_channel(&daqchan_vin);
    daq_delete_channel(&daqchan_cout);
    daq_delete_channel(&daqchan_cin);
    daq_exit();

    return ret;
}
