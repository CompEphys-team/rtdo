#ifndef FITINSPECTOR_H
#define FITINSPECTOR_H

#include <QWidget>
#include <QTableWidget>
#include "colorbutton.h"
#include "session.h"

namespace Ui {
class FitInspector;
}

class FitInspector : public QWidget
{
    Q_OBJECT

public:
    explicit FitInspector(Session &session, QWidget *parent = 0);
    ~FitInspector();

    void replot(bool summarising);

    struct Fit
    {
        QString label;
        QColor color, errColor;
        int idx;
        const GAFitter *gaf;
        const inline GAFitter::Output &fit() const { return gaf->results().at(idx); }
    };

    struct Group
    {
        QString label;
        QColor color;
        std::vector<Fit> fits;
    };

protected:
    std::vector<int> getSelectedRows(QTableWidget *table);
    ColorButton *getGraphColorBtn(int row);
    ColorButton *getErrorColorBtn(int row);
    ColorButton *getGroupColorBtn(int row);

private slots:
    void addGroup(std::vector<int> group = {}, QString label = "");
    void removeGroup();
    void on_saveGroups_clicked();
    void on_loadGroups_clicked();

    void updateFits();

private:
    Ui::FitInspector *ui;
    Session &session;

    std::vector<QColor> clipboard;

    std::vector<std::vector<int>> groups;
};

#endif // FITINSPECTOR_H
