#include "../include/readcsv.h"
#include <string>
#include <armadillo>
#include <fstream>
#include <sstream>
#include <iostream>
#include <utility>
#include <vector>
// holding info in a vector with pair(number, vector< 24x24 brightness values>)
CSV::CSV(const std::string& inputFile, int start, int end) {
  std::ifstream myfile(inputFile);
  if (!myfile.is_open()) {
    std::cerr << "Failed to open file: " << inputFile << std::endl;
  }
  std::string line;
  int lineNum = 0;
  while (std::getline(myfile, line)) {
    if (lineNum < start) {
      lineNum++;
      continue;
    }
    if (lineNum >= end) {
      break;
    }
    std::vector<int> row;
    std::stringstream ss(line);
    std::string cellValue;
    int correctNum;
      for (int j = 0; j < 785; j++) {
        std::getline(ss, cellValue, ',');
        if (j == 0) {
          correctNum = std::stoi(cellValue);
        }
        else {
          row.push_back(std::stoi(cellValue));
        }
      }
      data.push_back(std::make_pair(correctNum, row));
      lineNum++;
  }
  myfile.close();
}

std::vector<std::pair<int, std::vector<int>>> CSV::getData() {
  return data;
}
