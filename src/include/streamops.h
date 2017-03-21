#ifndef STREAMOPS_H
#define STREAMOPS_H

#include <QString>
#include <iostream>

std::istream &operator>>(std::istream &is, QString &str);
std::ostream &operator<<(std::ostream &os, const QString &str);


#endif // STREAMOPS_H
