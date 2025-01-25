/*
 * NetworkTrainer.hpp
 *
 *  Created on: 3 juli 2022
 *      Author: ruben
 */

#ifndef NETWORKTRAINER_HPP_
#define NETWORKTRAINER_HPP_

#include <vector>
#include <memory>

#include "Image.hpp"
#include "Network.hpp"

class NetworkTrainer {

private:
	std::vector<std::vector<double>> errorLayerVectors;
	std::vector<std::vector<double>> dC_db_layerVectors;
	std::vector<std::vector<std::vector<double>>> dC_dw_layerMatrices;
	std::vector<std::vector<double>> b_updateLayerVectors;
	std::vector<std::vector<std::vector<double>>> w_updateLayerMatrices;

	std::vector<std::vector<double>> u_layerVectors;
	std::vector<std::vector<std::vector<double>>> v_layerMatrices;
	std::vector<std::vector<double>> u_updateLayerVectors;
	std::vector<std::vector<std::vector<double>>> v_updateLayerMatrices;

	void calculateParameterUpdates(Network &net, double stepFactor);
	void calculateParameterUpdates(Network &net, double stepFactor, double weightDecayFactor);
	void calculateParameterUpdates(Network &net, double stepFactor, double weightDecayFactor, double frictionFactor);


	void resetUpdateData(Network &net);
	void zeroUpdateData(Network &net, unsigned int numberOfLayers);
	void zeroNeuronErrors(Network &net, unsigned int numberOfLayers);

public:
	void stochasticGradientDescent(Network &net,
			std::vector<std::unique_ptr<Image>> &trainingData,
			double learningRate, double weightDecayFactor, double frictionFactor,
			unsigned int miniBatchSize, unsigned int numberOfEpochs,
			const std::vector<std::unique_ptr<Image>> &validationData);

	void exampleBackPropagation(Network &net,
			const std::unique_ptr<Image> &trainingDatum);

};

#endif /* NETWORKTRAINER_HPP_ */
