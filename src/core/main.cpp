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
