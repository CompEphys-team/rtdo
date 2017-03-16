#include "mainwindow.h"
#include <QApplication>
#include "config.h"
#include <QFileDialog>

int main(int argc, char *argv[])
{
    setenv("GENN_PATH", LOCAL_GENN_PATH, 1);
    QApplication a(argc, argv);

    Config::init();

    MainWindow w;
    w.show();

    return a.exec();
}
