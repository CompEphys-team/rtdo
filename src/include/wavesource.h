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
        Deck = 3
    };

    Type type;
    size_t idx;
    Session *session;

    WaveSource() : session(nullptr) {}
    WaveSource(Session &session, Type type, size_t idx) : type(type), idx(idx), session(&session) {}

    const Wavegen::Archive &archive() const;
    const WavegenSelection *selection() const; //!< returns the nearest ancestor Selection, if any
    const WaveSubset *subset() const; //!< returns the nearest ancestor Subset, if any
    const WaveDeck *deck() const; //!< returns the deck, if type is Deck

    std::vector<Stimulation> stimulations() const;

    QString prettyName() const;
    int index() const; //!< Returns the overall index (eg. for comboboxes)

    friend QDataStream &operator<<(QDataStream &os, const WaveSource &);
    friend QDataStream &operator>>(QDataStream &is, WaveSource &);
};

Q_DECLARE_METATYPE(WaveSource)

#endif // WAVESOURCE_H
