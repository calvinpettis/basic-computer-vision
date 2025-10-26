#ifndef READCSV_H
#define READCSV_H
#include <string>
#include <armadillo>
#include <fstream>
#include <sstream>
#include <iostream>
#include <utility>
#include <vector>
class CSV {
public:
  std::vector<std::pair<int, std::vector<int>>> data;
  CSV(const std::string& inputFile, int start, int end);
  std::vector<std::pair<int, std::vector<int>>> getData();
};

#endif
