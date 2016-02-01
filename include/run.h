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

bool compile_model(XMLModel::outputType type);

bool run_vclamp(bool *stopFlag = 0);
bool run_wavegen(int focusParam = -1, bool *stopFlag = 0);

void run_digest(int generation, double best_err, double mavg, int nextS);


#endif // RUN_H
