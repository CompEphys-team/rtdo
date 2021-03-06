/** RTDO: Closed loop neuron model fitting
 *  Copyright (C) 2019 Felix Kern <kernfel+github@gmail.com>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
**/


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

struct RecStruct {
    std::vector<std::pair<size_t,size_t>> fit_coords;
    RegisterEntry *reg;
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

    void on_index_clicked();

    void on_validate_clicked();

    void on_finalise_clicked();

private:
    Ui::FitErrorPlotter *ui;
    Session *session;
    UniversalLibrary *lib = nullptr;

    bool summarising = false;
    std::vector<FitInspector::Group> data;

    std::map<std::pair<QString,QString>, RegisterEntry> register_map; // Keyed by pair<cell, protocol>
    std::vector<Protocol> protocols; // Protocols in register
    QDir register_dir;

    std::map<ResultKey, double> summaries;
    std::map<ResultKey, QVector<double>> traces;

    void push_run_pull(std::vector<ResultKey> keys, size_t keySz, bool get_traces);

    void plot_traces(Protocol &prot);
    void plot_boxes(std::vector<int> protocol_indices);

    std::vector<int> get_protocol_indices();
    int get_parameter_selection();
    bool loadRecording(RegisterEntry &reg, bool readData = true);

    std::vector<RecStruct> get_requested_recordings(int &nTraces, int &maxStimLen);
    void setup_lib_for_validation(int nTraces, int maxStimLen, bool get_traces, bool average_summary);
    std::vector<std::vector<std::vector<scalar>>> get_param_values(const GAFitter::Output &fit, int source);
    void crossvalidate(int target, std::vector<std::vector<std::vector<std::vector<std::vector<scalar>>>>> models = {});
    void validate(int target);
};

#endif // FITERRORPLOTTER_H
