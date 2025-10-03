#include <stdio.h>
#include <string>
#include <utility>
#include <vector>
#include "../include/readcsv.h"
int main(void) {
  std::string filepath = "../samples/mnist_train.csv";
  std::vector<std::pair<int, std::vector<int>>> dataFromCSV;
  CSV test(filepath, 1);
  dataFromCSV = test.getData();
  printf("Number that this data represents: %d\n", dataFromCSV[0].first);
  for (int i = 1; i < 784; i++) {
    printf("%d ", dataFromCSV[0].second[i]);
  }
}
