#ifndef NETWORK_H

#define NETWORK_H
#include <armadillo>
class Network {
public:
  int num_layers;
  std::vector<int> sizes;
  std::vector<arma::mat> biases;
  std::vector<arma::mat> weights;

  Network(std::vector<int> size);
  arma::mat sigmoid(arma::mat z);
  arma::mat feedforward(arma::mat a);
void stochastic_gradient_descent(std::vector<std::pair<int, std::vector<int>>> training_data, int epochs, int mini_batch_size, int eta, std::vector<std::pair<int, std::vector<int>>> test_data = {});
};
//void update_mini_batch(std::vector<std::pair<int, std::vector<int>>> mini_batch, int eta);


#endif
