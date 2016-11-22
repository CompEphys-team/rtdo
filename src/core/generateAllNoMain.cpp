// GeNN's generateALL.cc has an int main(int, char*[]), because it's meant to be compiled and run separately.
// This just allows us to compile it into the main project without conflict.
#define main generateAll

// MODEL is included in generateALL.cc; in standard GeNN use, this is a code file with the model definition,
// passed in through genn-buildmodel.sh
#define MODEL "metamodel.h"

#include "lib/src/generateALL.cc"
