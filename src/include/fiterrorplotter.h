#ifndef FITERRORPLOTTER_H
#define FITERRORPLOTTER_H

#include <QWidget>
#include "fitinspector.h"

namespace Ui {
class FitErrorPlotter;
}

struct Protocol {
    QString name;
    double dt;
    int blankCycles;
    std::vector<Stimulation> stims;
    std::vector<iStimulation> istims;
    std::vector<iObservations> iObs;
    int idx;
};

struct RegisterEntry {
    QString file, cell, protocol;
    RunData rund;
    QVector<QVector<double>> data;
    Protocol *pprotocol;
    CannedDAQ::ChannelAssociation assoc;
};

class FitErrorPlotter : public QWidget
{
    Q_OBJECT

    /** ResultKey uniquely identifies a comparison by:
     * - Protocol index, local to the register used
     * - Stimulation index, local to the protocol
     * - Global fit index (cf FitInspector::Fit::idx)
     * - Parameter selection (cf parameter combobox in the UI; -2=target, -1=Final, N(>=0)=Epoch N
     */
    using ResultKey = std::tuple<int,int,int,int>;

public:
    explicit FitErrorPlotter(QWidget *parent = 0);
    ~FitErrorPlotter();

    void init(Session *session);

    void setData(std::vector<FitInspector::Group> data, bool summarising);

    void replot();

private slots:
    void on_register_path_textChanged(const QString &arg1);

    void on_run_clicked();

    void on_pdf_clicked();

private:
    Ui::FitErrorPlotter *ui;
    Session *session;
    UniversalLibrary *lib = nullptr;

    bool summarising = false;
    std::vector<FitInspector::Group> data;

    std::map<std::pair<QString,QString>, RegisterEntry> register_map; // Keyed by pair<cell, protocol>
    std::vector<Protocol> protocols; // Protocols in register
    QDir register_dir;

    //                       MSE summary, current trace
    std::map<ResultKey, std::pair<double, QVector<double>>> results;

    void push_run_pull(std::vector<ResultKey> keys, size_t keySz);

    void plot_traces(Protocol &prot);
    void plot_boxes(std::vector<int> protocol_indices);

    std::vector<int> get_protocol_indices();
    int get_parameter_selection();
    bool loadRecording(RegisterEntry &reg, bool readData = true);
};

#endif // FITERRORPLOTTER_H