#ifndef WAVESOURCE_H
#define WAVESOURCE_H

#include "wavegenselector.h"

class WaveSubset;
class WaveDeck;

class WaveSource
{
public:
    enum Type {
        Archive = 0,
        Selection = 1,
        Subset = 2,
        Deck = 3,
        Manual = 4
    };

    Type type;
    size_t idx;
    int waveno;
    Session *session;

    WaveSource() : waveno(-1), session(nullptr) {}
    WaveSource(Session &session, Type type, size_t idx, int waveno = -1) : type(type), idx(idx), waveno(waveno), session(&session) {}

    const Wavegen::Archive *archive() const;
    const WavegenSelection *selection() const; //!< returns the nearest ancestor Selection, if any
    const WaveSubset *subset() const; //!< returns the nearest ancestor Subset, if any
    const WaveDeck *deck() const; //!< returns the deck, if type is Deck

    int resultIndex() const;

    std::vector<Stimulation> stimulations() const;
    std::vector<MAPElite> elites() const;
    std::vector<iStimulation> iStimulations(double dt) const;

    QString prettyName() const;
    int index() const; //!< Returns the overall index (eg. for comboboxes), ignoring waveno.

    friend QDataStream &operator<<(QDataStream &os, const WaveSource &);
    friend QDataStream &operator>>(QDataStream &is, WaveSource &);
    friend bool operator==(const WaveSource &lhs, const WaveSource &rhs);

    static constexpr quint32 version = 100;
};

Q_DECLARE_METATYPE(WaveSource)

#endif // WAVESOURCE_H
