#include "../include/network.h"
#include <algorithm>
#include <random>
#include <vector>
#include <armadillo>
#include <bits/stdc++.h>
using namespace std;
//using namespace arma;
Network::Network(vector<int> size) {
  sizes = size;
  num_layers = sizes.size();
  int j = sizes.size() - 1;
  for (int i = 1; i < sizes.size(); i++) {
    arma::mat b = arma::randn(sizes[i], 1);
    biases.push_back(b);
    arma::mat c = arma::randn(sizes[i], sizes[i - 1]);
    weights.push_back(c);
  }
}
arma::mat Network::sigmoid(arma::mat z) {
  return 1.0f/(1.0f+exp(-z));
}
arma::mat Network::feedforward(arma::mat a) {
  for (int i = 0; i < weights.size(); i++) {
  arma::mat b = biases[i];
  arma::mat w = weights[i];
  a = sigmoid((w * a)+b);
  }
  return a;
}
void Network::stochastic_gradient_descent(std::vector<std::pair<int, std::vector<int>>> training_data, int epochs, int mini_batch_size, int eta, std::vector<std::pair<int, std::vector<int>>> test_data) {
  int n_test, n;
  std::vector<std::pair<int, std::vector<int>>> mini_batches;
  if (!test_data.empty()) {
    n_test = test_data.size();
    n = training_data.size();
  }
  random_device rand;
  mt19937 randSeed(rand());
  for (int i = 0; i < epochs; i++) {
    shuffle(training_data.begin(), training_data.end(), randSeed);
    for (int j = 0; j < mini_batch_size; j++) {
      mini_batches.push_back(training_data[j]);
    }
    for (int k = 0; k < mini_batches.size(); k++) {
      //Network::update_mini_batch(mini_batches[k]);
    }
    if (!test_data.empty()) {
      //printf("Epoch {%d}: {%d} / {%d}\n", i, Network::evaluate(test_data), n_test);
    }
    else {
      printf("Epoch {%d} complete", i);
    }
  }
}

// void Network::update_mini_batch(std::vector<std::pair<int, std::vector<int>>> mini_batch, int eta) {
//   
// }
