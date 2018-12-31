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
        QCoreApplication a(argc, argv);
        std::ifstream is(argv[1]);
        if ( !is.good() ) {
            std::cerr << "Failed to open loader file \"" << argv[1] << "\"." << std::endl;
            return -1;
        }
        std::string head, dopfile, sessdir;
        is >> head;
        if ( head != "#project" ) {
            std::cerr << "Unexpected \"" << head << "\" on line 1 (expected \"#project\")." << std::endl;
            return -2;
        }
        std::getline(is, dopfile);
        QString dopfile_trimmed = QString::fromStdString(dopfile).trimmed();
        QFileInfo dopfile_info(dopfile_trimmed);
        if ( !dopfile_info.exists() || !dopfile_info.isReadable() ) {
            std::cerr << "Failed to access project file \"" << dopfile_trimmed << "\"." << std::endl;
            return -3;
        }

        Project project(dopfile_trimmed);

        is >> head;
        if ( head != "#session" ) {
            std::cerr << "Unexpected \"" << head << "\" on line 2 (expected \"#session\")." << std::endl;
            return -4;
        }
        std::getline(is, sessdir);
        QString sessdir_trimmed = QString::fromStdString(sessdir).trimmed();
        QFileInfo sessdir_info(sessdir_trimmed);
        if ( !sessdir_info.exists() || !sessdir_info.isDir() ) {
            std::cerr << "Failed to access session directory \"" << sessdir_trimmed << "\"." << std::endl;
            return -5;
        }

        Session session(project, sessdir_trimmed);

        QObject::connect(&session, &Session::dispatchComplete, &a, &QCoreApplication::quit);

        session.exec_desiccated(QString(argv[1]));

        return a.exec();
    }
}
