#ifndef WAVESOURCE_H
#define WAVESOURCE_H

#include "wavegenselector.h"

class WaveSource
{
public:
    enum Type {
        Archive = 0,
        Selection = 1
    };

    Type type;
    size_t idx;
    Session *session;

    WaveSource() : session(nullptr) {}
    WaveSource(Session &session, Type type, size_t idx) : type(type), idx(idx), session(&session) {}

    const Wavegen::Archive &archive() const;
    const WavegenSelection *selection() const; //!< returns null if the type doesn't have a Selection
    QString prettyName() const;
    int index() const; //!< Returns the overall index (eg. for comboboxes)

    friend QDataStream &operator<<(QDataStream &os, const WaveSource &);
    friend QDataStream &operator>>(QDataStream &is, WaveSource &);
};

Q_DECLARE_METATYPE(WaveSource)

#endif // WAVESOURCE_H
