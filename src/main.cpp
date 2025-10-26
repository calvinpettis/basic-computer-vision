#include <stdio.h>
#include <string>
#include <utility>
#include <vector>
#include "../include/readcsv.h"
#include "../include/network.h"
int main(void) {
  std::string filepath = "../samples/mnist_train.csv";
  std::vector<std::pair<int, std::vector<int>>> training_data;
  std::vector<std::pair<int, std::vector<int>>> test_data;
  CSV train(filepath, 0, 1000);
  training_data = train.getData();
  CSV test(filepath, 1001, 1500);
  test_data = test.getData();
  Network net({784,30,10});
  net.stochastic_gradient_descent(training_data, 30, 10, 1.0, test_data);
  // printf("Number that this data represents: %d\n", dataFromCSV[0].first);
  // for (int i = 1; i < 784; i++) {
  //   printf("%d ", dataFromCSV[0].second[i]);
  // }
}
