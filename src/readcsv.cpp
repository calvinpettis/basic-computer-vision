#include "../include/readcsv.h"
#include <string>
#include <armadillo>
#include <fstream>
#include <sstream>
#include <iostream>
#include <utility>
#include <vector>
// holding info in a vector with pair(number, vector< 24x24 brightness values>)
CSV::CSV(const std::string& inputFile, int numsToRead) {
  std::ifstream myfile(inputFile);
  if (!myfile.is_open()) {
    std::cerr << "Failed to open file: " << inputFile << std::endl;
  }
  std::string line;
  while (std::getline(myfile, line)) {
    std::vector<int> row;
    std::stringstream ss(line);
    std::string cellValue;
    int correctNum;
    for (int i = 0; i < numsToRead; i++) {
      for (int j = 0; j < 785; j++) {
        if (j == 0) {
          std::getline(ss, cellValue, ',');
          correctNum = std::stoi(cellValue);
        }
        else {
          std::getline(ss, cellValue, ',');
          row.push_back(std::stoi(cellValue));
        }
      }
      data.push_back(std::make_pair(correctNum, row));
    }
    myfile.close();
  }

}

std::vector<std::pair<int, std::vector<int>>> CSV::getData() {
  return data;
}
