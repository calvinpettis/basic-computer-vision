#ifndef NETWORK_H

#define NETWORK_H
#include <armadillo>
#include <utility>
class Network {
public:
  int num_layers;

  std::vector<int> sizes;

  std::vector<arma::mat> biases;

  std::vector<arma::mat> weights;

  Network(std::vector<int> size);

  arma::mat feedforward(arma::mat a);

  void stochastic_gradient_descent(std::vector<std::pair<int, std::vector<double>>>& training_data, int epochs, int mini_batch_size, float eta, std::vector<std::pair<int, std::vector<double>>> test_data = {});

  void update_mini_batch(std::vector<std::pair<int, std::vector<double>>> mini_batch, float eta);

  std::pair<std::vector<arma::mat>,std::vector<arma::mat>> backprop(std::vector<double>& x, int y);

  int evaluate(std::vector<std::pair<int, std::vector<double>>>& test_data);

arma::mat cost_derivative(arma::vec output_activation, int y);

  arma::mat sigmoid(arma::mat z);

  arma::mat sigmoid_prime(arma::mat z);

};
#endif
