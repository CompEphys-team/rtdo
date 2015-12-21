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

daq_channel daqchan_vout = DAQ_CHANNEL_INIT;
daq_channel daqchan_cout = DAQ_CHANNEL_INIT;
daq_channel daqchan_vin = DAQ_CHANNEL_INIT;
daq_channel daqchan_cin = DAQ_CHANNEL_INIT;
struct _sim_params sim_params = SIMPARAM_DEFAULT;

int main(int argc, char *argv[])
{
    int ret=0, have_calibration;
    std::string device = "/dev/comedi0";
    std::string cal = "/home/felix/projects/rtdo/ni6251.calibrate";
    sim_params.outdir = "/home/felix/projects/build/rtdo/output";
    sim_params.sigfile = "/home/felix/projects/rtdo/models/Lymnaea_B1.dat";
    sim_params.vc_wavefile = "/home/felix/projects/rtdo/vclamp/wave2.dat";
    sim_params.modelfile = "/home/felix/projects/rtdo/models/Lymnaea_B1.cc";

    //--------------------------------------------------------------
    // Set up softrt-daq and channels
    if ( (ret = daq_load_lib("libcomedi.so")) )
        return ret;
    if ( (ret = daq_open_device(device.c_str())) )
        return ret;
    have_calibration = !daq_load_calibration(cal.c_str());

    daqchan_vout.type = COMEDI_SUBD_AO;
    daqchan_vout.subdevice = daq_get_subdevice(daqchan_vout.type, 0);
    daqchan_vout.gain = 20.0;
    if ( (ret = daq_create_converter(&daqchan_vout)) )
        return ret;

    daqchan_cout.type = COMEDI_SUBD_AO;
    daqchan_cout.channel = 1;
    daqchan_cout.subdevice = daq_get_subdevice(daqchan_cout.type, 0);
    if ( (ret = daq_create_converter(&daqchan_cout)) )
        return ret;

    daqchan_vin.type = COMEDI_SUBD_AI;
    daqchan_vin.aref = AREF_DIFF;
    daqchan_vin.subdevice = daq_get_subdevice(daqchan_vin.type, 0);
    daqchan_vin.gain = 10.0;
    if ( (ret = daq_create_converter(&daqchan_vin)) )
        return ret;

    daqchan_cin.type = COMEDI_SUBD_AI;
    daqchan_cin.aref = AREF_DIFF;
    daqchan_cin.channel = 1;
    daqchan_cin.subdevice = daq_get_subdevice(daqchan_cin.type, 0);
    daqchan_cin.gain = 10.0;
    if ( (ret = daq_create_converter(&daqchan_cin)) )
        return ret;

    //--------------------------------------------------------------
    // Set up RT
    if ( (ret = rtdo_init("/dev/comedi0")) )
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
    if ( have_calibration )
        daq_unload_calibration();
    daq_close_device();
    daq_unload_lib();

    return ret;
}
