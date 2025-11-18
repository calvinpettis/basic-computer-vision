#include <stdio.h>
#include <string>
#include <utility>
#include <vector>
#include "../include/readcsv.h"
#include "../include/network.h"
int main(void) {
  //digit char dataset is 339,337 train 103,113 test
  // best hyperparameters so far: 
  // one hidden layer, 1.0 learning rate, batch size 64
  printf("Welcome to Cal's handwritten text recognizer! Are you training or loading a model?\nPlease type in either:\n\"train\" or \"load\".\n");
  std::string response;
  char tmp[10];
  scanf("%s", tmp);
  response = tmp;
  Network net({784,100,100,36});
  if (response == "train") {
    printf("Train model with testing data?\nEnter y or n\n");
    std::string filepath = "../samples/digit_char_dataset.csv";
    std::vector<std::pair<int, std::vector<double>>> training_data;
    std::vector<std::pair<int, std::vector<double>>> test_data;
    CSV train(filepath, 0, 339337);
    training_data = train.getData();
    CSV test(filepath, 339338, 439338);
    test_data = test.getData();
    net.stochastic_gradient_descent(training_data, 100, 256, 0.1, test_data);
  }
  else if (response == "load") {
    net.load_model();
    printf("Enter filename of image for text recognition\n");
    char tmp[100];
    scanf("%s", tmp);
    std::string filename = tmp;

  }
  // printf("Number that this data represents: %d\n", dataFromCSV[0].first);
  // for (int i = 1; i < 784; i++) {
  //   printf("%d ", dataFromCSV[0].second[i]);
  // }
}
