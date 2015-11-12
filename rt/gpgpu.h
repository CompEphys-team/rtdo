#include <linux/comedi.h>

void gpu_init(int size);
void gpu_exit();
void gpu_process(lsampl_t *in, lsampl_t *out, lsampl_t max);
