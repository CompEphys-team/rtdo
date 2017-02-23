#include "mainwindow.h"
#include <QApplication>
#include "config.h"
#include <QFileDialog>

int main(int argc, char *argv[])
{
    setenv("GENN_PATH", LOCAL_GENN_PATH, 1);
    QApplication a(argc, argv);

    Config::init();

    std::string fname = QFileDialog::getOpenFileName().toStdString();
    std::string dir = QFileDialog::getExistingDirectory().toStdString();
    if ( fname.empty() || dir.empty() )
        return -1;
    Config::Model.filepath = fname;
    Config::Model.dirpath = dir;

    MainWindow w;
    w.show();

    return a.exec();
}
