#include "mainwindow.h"
#include <QApplication>
#include <fstream>
#include <QFileInfo>

int main(int argc, char *argv[])
{
    setenv("GENN_PATH", LOCAL_GENN_PATH, 1);
    if ( argc == 1 ) {
        QApplication a(argc, argv);

        MainWindow w;
        w.show();

        return a.exec();
    } else {
        std::ifstream loader_file(argv[1]);
        if ( !loader_file.good() ) {
            std::cerr << "Failed to open loader file \"" << argv[1] << "\"." << std::endl;
            return;
        }
        std::string head, dopfile, sessdir;
        is >> head;
        if ( head != "#project" ) {
            std::cerr << "Unexpected \"" << head << "\" on line 1 (expected \"#project\")." << std::endl;
            return;
        }
        std::getline(is, dopfile);
        QFileInfo dopfile_info(QString::fromStdString(dopfile));
        if ( !dopfile_info.exists() || !dopfile_info.isReadable() ) {
            std::cerr << "Failed to access project file \"" << dopfile << "\"." << std::endl;
            return;
        }

        Project project(QString::fromStdString(dopfile));

        is >> head;
        if ( head != "#session" ) {
            std::cerr << "Unexpected \"" << head << "\" on line 2 (expected \"#session\")." << std::endl;
            return;
        }
        std::getline(is, sessdir);
        QFileInfo sessdir_info(QString::fromStdString(sessdir));
        if ( !sessdir_info.exists() || !sessdir_info.isDir() ) {
            std::cerr << "Failed to access session directory \"" << dopfile << "\"." << std::endl;
            return;
        }

        Session session(project, QString::fromStdString(sessdir));

        session.exec_desiccated(is);
        return 0;
    }
}
