#ifndef CANNEDDAQ_H
#define CANNEDDAQ_H

#include "daq.h"
#include <vector>
#include "wavesubsets.h"

class CannedDAQ : public DAQ
{
public:
    CannedDAQ(Session &s);
    ~CannedDAQ() = default;

    double getAdjustableParam(size_t);
    void setAdjustableParam(size_t, double) {}
    int throttledFor(const Stimulation &) { return 0; }
    void run(Stimulation s, double settleDuration = 0);
    void next();
    void reset();

    void setRecord(std::vector<Stimulation> stims, QString record, bool readData = true, bool useQueuedSettings = false);
    std::vector<QuotedString> channelNames;

    struct ChannelAssociation {
        int Iidx, Vidx, V2idx;
        double Iscale, Vscale, V2scale;
    };
    static ChannelAssociation s_assoc; //!< UX legacy - not used internally
    ChannelAssociation assoc; //!< This one is (during data reading in setRecord).

    void getSampleNumbers(const std::vector<Stimulation> &stims, double dt,
                          int *nTotal, int *nBuffer = nullptr, int *nSamples = nullptr);

protected:
    int prepareRecords(std::vector<Stimulation> stims, bool useQueuedSettings);

    struct Record
    {
        Stimulation stim;
        int nBuffer, nTotal;
        std::vector<double> I, V, V2;
    };
    std::vector<Record> records;
    size_t currentRecord;
    int recordIndex;
    double settleDuration;

    void getReferenceParams(QString record);
    std::vector<double> ref_params;
};

#endif // CANNEDDAQ_H
