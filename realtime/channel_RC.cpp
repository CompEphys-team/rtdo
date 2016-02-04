/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-26

--------------------------------------------------------------------------*/
#ifdef CONFIG_RT

#include "channel_impl.h"
#include "RC_wrapper.h"

// ---------------------------------------- Public Channel API -----------------------------------------------

bool Channel::read(lsampl_t &sample, bool hint) const
{
    if ( hint ) {
        RC_comedi_data_read_hint(pImpl->_deviceRC, pImpl->_subdevice,
                                 pImpl->_channel, pImpl->_range, pImpl->_aref);
    }
    return (1 == RC_comedi_data_read(pImpl->_deviceRC, pImpl->_subdevice,
                                     pImpl->_channel, pImpl->_range, pImpl->_aref, &sample));
}

bool Channel::write(lsampl_t sample) const
{
    return (1 == RC_comedi_data_write(pImpl->_deviceRC, pImpl->_subdevice,
                                      pImpl->_channel, pImpl->_range, pImpl->_aref, sample));
}

#endif
