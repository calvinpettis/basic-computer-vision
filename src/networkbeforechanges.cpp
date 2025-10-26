#include "../include/network.h"
#include <algorithm>
#include <iterator>
#include <random>
#include <utility>
#include <vector>
#include <armadillo>
#include <bits/stdc++.h>
//using namespace std;
//using namespace arma;

/* 
 * C++ implementation of Michael Neilsen's starter code
 * A simple neural network implementing a computer vision model for 28x28px images for handwritten digits.
 * (Using the MNIST dataset, converted to 0-255 greyscale values in CSV format)
 * @param: size, a list containing the amounts of neurons in each layer
 * Ex: Network([784, 100, 10] = 3 layers, 784 input neurons (28x28), 100 hidden neurons, 10 output neurons (0-9))  
 *
*/
Network::Network(std::vector<int> size) {
  sizes = size;
  num_layers = sizes.size();
  int j = sizes.size() - 1;
  //we dont want weights and biases for input layer of neurons, so start with second item in array
  for (int i = 1; i < sizes.size(); i++) {
    //matrices of n biases for each neuron of each layer
    arma::mat b = arma::randn(sizes[i], 1);
    biases.push_back(b);
    //matrices of the weights connecting two layers
    arma::mat c = arma::randn(sizes[i], sizes[i - 1]);
    weights.push_back(c);
  }
}
/*
 * Apply the sigmoid function to the neural network
 * @param a
 */
arma::mat Network::feedforward(arma::mat a) {
  for (int i = 0; i < weights.size(); i++) {
    arma::mat b = biases[i];
    arma::mat w = weights[i];
    a = sigmoid((w * a)+b);
  }
  return a;
}
/*
 * Compute gradient, then move in the opposite direction to minimize the cost function (simulating the "ball rolling down the hill")
 * @param training_data, the subset of the MNIST dataset to train on.
 * @param epochs, the number of training steps to take
 * @param mini_batch_size, the number of training data points for one mini batch
 * @param eta, the 'learning rate'
 * @param test_data, the data to validate training on
 * @return none
 */
void Network::stochastic_gradient_descent (std::vector<std::pair<int, std::vector<int>>>& training_data, int epochs, int mini_batch_size, float eta, std::vector<std::pair<int, std::vector<int>>> test_data) {
  int n_test, n;
  std::vector<std::pair<int, std::vector<int>>> mini_batches;
  if (!test_data.empty()) {
    n_test = test_data.size();
  }
  n = training_data.size();
  std::random_device rand;
  std::mt19937 randSeed(rand());
  for (int i = 0; i < epochs; i++) {
    shuffle(training_data.begin(), training_data.end(), randSeed);
    for (int j = 0; j < mini_batch_size; j++) {
      mini_batches.push_back(training_data[j]);
    }
    for (int k = 0; k < mini_batches.size(); k++) {
      Network::update_mini_batch(mini_batches[k]);
    }
    if (!test_data.empty()) {
      printf("Epoch {%d}: {%d} / {%d}\n", i, Network::evaluate(test_data), n_test);
    }
    else {
      printf("Epoch {%d} complete", i);
    }
  }
}
/*
 * Mini batch breaks up the training input into small "batches", training until input is exhausted 
 * @param mini_batch, the subset of training data
 * @param eta, the learning rate
 */
void Network::update_mini_batch(std::vector<std::pair<int, std::vector<int>>>& mini_batch, float eta) {
  std::vector<arma::mat> nabla_b = biases;
  std::vector<arma::mat> nabla_w = weights;
  for (int i = 0; i < biases.size(); i++) {
    nabla_b[i] = biases[i].zeros();
  }
  for (int i = 0; i < weights.size(); i++) {
    nabla_w[i] = weights[i].zeros();
  }
  arma::mat w;
  arma::mat b;
  arma::mat nb;
  arma::mat nw;
  arma::mat dnb;
  arma::mat dnw;
  for (int i = 0; i < mini_batch.size(); i++) {
    int y = mini_batch[i].first;
    std::vector<int> x = mini_batch[i].second;
    auto result = backprop(x, y);
    std::vector<arma::mat> delta_nabla_b = result.first;
    std::vector<arma::mat> delta_nabla_w = result.second;
    for (int j = 0; j < nabla_b.size(); j++) {
      nb = nabla_b[j];
      dnb = delta_nabla_b[j];
      nabla_b[j] = nb + dnb;
    }
    for (int k = 0; k < nabla_w.size(); k++) {
      nw = nabla_w[k];
      dnw = delta_nabla_w[k];
      nabla_w[k] = nw + dnw;
    }
  }
  for (int i = 0; i < weights.size(); i++) {
    w = weights[i];
    nw = nabla_w[i];
    weights[i] = (w-(eta/mini_batch.size()) * nw);
  }
  for (int i = 0; i < biases.size(); i++) {
    b = biases[i];
    nb = nabla_b[i];
    biases[i] = (b-(eta/mini_batch.size()) * nb);
  }
}
/*
* 
*/
std::pair<std::vector<arma::mat>,std::vector<arma::mat>> Network::backprop(std::vector<int>& x, int y) {
  std::vector<arma::mat> nabla_b = biases;
  std::vector<arma::mat> nabla_w = weights;
  for (int i = 0; i < biases.size(); i++) {
    nabla_b[i] = biases[i].zeros();
  }
  for (int i = 0; i < weights.size(); i++) {
    nabla_w[i] = weights[i].zeros();
  }
  arma::vec activation = arma::conv_to<arma::vec>::from(x);
  std::vector<arma::vec> activations = {arma::conv_to<arma::vec>::from(x)};
  std::vector<arma::vec> zs = {};
  for (int i = 0; i < biases.size(); i++) {
    arma::mat b = biases[i];
    arma::mat w = weights[i];
    arma::mat z = (w * activation) + b;
    zs.push_back(z);
    activation = Network::sigmoid(z);
    activations.push_back(activation);
  }
  std::vector<arma::mat> delta = Network::cost_derivative(activations[activations.size() - 1], y) * sigmoid_prime(zs[zs.size() - 1])
  nabla_b[nabla_b.size() - 1] = delta;
  nabla_w[nabla_w.size() - 1] = delta * arma::trans(activations[activations.size() - 2]);
  for (int l = 2; l < num_layers; l++) {
    arma::mat z = zs[zs.size() - l];
    sp = sigmoid_prime(z);
    delta = (arma::trans(weights[weights.size() - l + 1]) * delta) * sp;
    nabla_b[nabla_b.size() - l] = delta;
    nabla_w[nabla_w.size() - l] = delta * arma::trans(activations[activations.size() - l - 1]);
  }
  return make_pair(nabla_b, nabla_w);
}
/*
 *
 */
int Network::evaluate(std::vector<std::pair<int, std::vector<int>>>& test_data) {
 //       <our num, correct num> 
  std::vector<std::pair<int,int>> test_results;

  for (int i = 0; i < test_data.size(); i++) {
    arma::mat result = arma::conv_to<arma::mat>::from(test_data[i].second);
    result = Network::feedforward(result);
    int x = result.index_max();
    test_results.push_back(std::make_pair(x, test_data[i].first));
  }
  int sum = 0;
  for (int i = 0; i < test_results.size(); i++) {
    if (test_results[i].first == test_results[i].second) {
      sum++;
    }
  }
  return sum;
}


std::vector<arma::mat> Network::cost_derivative(arma::vec output_activation, int y) {
  std::vector<arma::mat> vec;
  for (int i = 0; i < output_activation.size(); i++) {
    output_activation[i] -= y;
  }
  vec.push_back(output_activation);
  return vec;
}
/*
* Normalize the sum of the matrix to something non linear 
*/
arma::mat Network::sigmoid(arma::mat z) {
  return 1.0f/(1.0f+exp(-z));
}

arma::mat Network::sigmoid_prime(arma::mat z) {
  return Network::sigmoid(z)*(1-Network::sigmoid(z));
}
