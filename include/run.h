/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2015-12-08

--------------------------------------------------------------------------*/

#ifndef RUN_H
#define RUN_H

#include "types.h"
#include "rt.h"
#include <iostream>
#include <queue>
#include <vector>
#include "xmlmodel.h"

typedef struct {
  double t;
  double ot;
  double dur;
  double baseV;
  int N;
  std::vector<double> st;
  std::vector<double> V;
  double fit;
} inputSpec;
std::ostream &operator<<(std::ostream &os, inputSpec &I);

int compile_model(XMLModel::outputType type);

bool run_vclamp_start();
void run_vclamp_stop();

daq_channel *run_get_active_outchan();
daq_channel *run_get_active_inchan();

void run_digest(double best_err, double mavg, int nextS);
int run_check_break();

double run_getsample(float t);
void run_setstimulus(inputSpec I);

bool run_wavegen_start(int focusParam = -1);
void run_wavegen_stop();

#endif // RUN_H
