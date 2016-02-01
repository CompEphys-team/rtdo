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

#include "xmlmodel.h"
#include "shared.h"

int compile_model(XMLModel::outputType type);

bool run_vclamp_start();
void run_vclamp_stop();

void run_digest(int generation, double best_err, double mavg, int nextS);
void run_use_backlog(backlog::Backlog *log);
int run_check_break();

double run_getsample(float t);
void run_setstimulus(inputSpec I);

bool run_wavegen_start(int focusParam = -1);
void run_wavegen_stop();

#endif // RUN_H
