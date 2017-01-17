#ifndef SUPPORTCODE_H
#define SUPPORTCODE_H

#include <cuda_runtime.h>
#include "types.h"

__host__ __device__ scalar getCommandVoltage(const Stimulation &I, scalar t);

#endif // SUPPORTCODE_H
