/*
 * NetworkTrainer.cpp
 *
 *  Created on: 3 juli 2022
 *      Author: ruben
 */

#include "NetworkTrainer.hpp"

#include <vector>
#include <memory>
#include <random>
#include <iostream>

#include <SFML/Graphics.hpp>

#include "NetworkTester.hpp"
#include "QuadraticCoster.hpp"
#include "CrossEntropyCoster.hpp"

void NetworkTrainer::stochasticGradientDescent(Network &net,
		std::vector<std::unique_ptr<Image> > &trainingData,
		const double learningRate, double weightDecayFactor, double frictionFactor,
		const unsigned int miniBatchSize, const unsigned int numberOfEpochs,
		const std::vector<std::unique_ptr<Image>> &validationData) {

	// Reset all update data
	resetUpdateData(net);

	// Train the network
	auto numberOfMiniBatches = static_cast<unsigned int>(trainingData.size()
			/ miniBatchSize);
	double stepSize = -learningRate / miniBatchSize;

	std::random_device rd;
	std::mt19937 g(rd());

	// go through all epochs
	for (unsigned int e = 0; e < numberOfEpochs; ++e) {
		// shuffle the examples
		std::shuffle(trainingData.begin(), trainingData.end(), g);

		// go through the examples, one minibatch at a time
		for (unsigned int b = 0; b < numberOfMiniBatches; ++b) {
			const int start = b * miniBatchSize;
			const int end = (b + 1) * miniBatchSize;

			// reset updates to weights and biases (applied before each minibatch is handled)
			const unsigned int numberOfLayers = net.getNumberOfLayers();
			zeroUpdateData(net, numberOfLayers);

			// go through all training examples in the current minibatch
			for (int i = start; i < end; ++i) {

				// zero the neuron errors (applied before each example is handled)
				zeroNeuronErrors(net, numberOfLayers);

				// get partial derivatives of the cost function with respect to the parameters
				exampleBackPropagation(net, trainingData[i]);
			}

			// calculate parameter updates
			calculateParameterUpdates(net, stepSize, weightDecayFactor, frictionFactor);

			// update the weights and biases
			net.updateWeightsAndBiases(b_updateLayerVectors,
					w_updateLayerMatrices);
		}
		std::cout << "epoch: " << e << std::endl;

		std::cout << " - correct (validation): "
				<< NetworkTester::testNetwork(net, validationData) << std::endl;

//		double cost = CrossEntropyCoster::calculateTotalCost(net, trainingData);
//		std::cout << " - cost (crossEntropy): " << cost << std::endl;
	}
}

void NetworkTrainer::exampleBackPropagation(Network &net,
		const std::unique_ptr<Image> &trainingDatum) {

	const unsigned int numberOfLayers = net.getNumberOfLayers();
	unsigned int L = numberOfLayers - 1;

	// feed forward to obtain the actual outputs of this example
	auto &outputActivations = net.feedForward(
			trainingDatum->getInputActivations());

	// calculate the errors in the output layer
	const unsigned int neuronsInLastLayer = net.getNeuronsInLayer()[L];

	for (unsigned int j = 0; j < neuronsInLastLayer; ++j) {
		auto a = outputActivations[j];
		auto y = trainingDatum->getDesiredOutputActivations()[j];
		errorLayerVectors[L][j] =
				CrossEntropyCoster::calculateExampleErrorInNeuron(a, y);

//		auto z = net.getActivationInputVectorOfLayer()[L][j];
//		m_errorVectorOfLayer[L][j] =
//				QuadraticCoster::calculateExampleErrorInNeuron(a, y, z);
	}

	// backpropagate to calculate the errors in all other layers (except the input layer)
	for (unsigned int l = L - 1; l > 0; --l) {

		const unsigned int numberOfNeuronsK = net.getNeuronsInLayer()[l];

		for (unsigned int k = 0; k < numberOfNeuronsK; ++k) {

			const unsigned int numberOfNeuronsJ = net.getNeuronsInLayer()[l + 1];

			for (unsigned int j = 0; j < numberOfNeuronsJ; ++j) {
				errorLayerVectors[l][k] +=
						net.getWeightMatrixOfLayer()[l + 1][j][k]
								* errorLayerVectors[l + 1][j];
			}
			errorLayerVectors[l][k] *= net.dAF_dz(
					net.getActivationInputVectorOfLayer()[l][k]);
		}
	}

	// calculate partial derivatives of the cost function with respect to all the parameters
	for (unsigned int l = 1; l < numberOfLayers; ++l) {

		const unsigned int numberOfNeuronsJ = net.getNeuronsInLayer()[l];

		for (unsigned int j = 0; j < numberOfNeuronsJ; ++j) {
			const double error = errorLayerVectors[l][j];
			dC_db_layerVectors[l][j] += error;

			const unsigned int numberOfNeuronsK = net.getNeuronsInLayer()[l - 1];
			const auto &activationVector = net.getActivationVectorOfLayer()[l
					- 1];

			for (unsigned int k = 0; k < numberOfNeuronsK; ++k) {
				dC_dw_layerMatrices[l][j][k] += activationVector[k] * error;
			}
		}
	}

}

void NetworkTrainer::calculateParameterUpdates(Network &net,
		double stepFactor) {
	const unsigned int numberOfLayers = net.getNumberOfLayers();

	for (unsigned int l = 1; l < numberOfLayers; ++l) {

		const unsigned int j_neuronCount = net.getNeuronsInLayer()[l];

		for (unsigned int j = 0; j < j_neuronCount; ++j) {

			const unsigned int k_neuronCount = net.getNeuronsInLayer()[l - 1];

			b_updateLayerVectors[l][j] = stepFactor * dC_db_layerVectors[l][j];

			for (unsigned int k = 0; k < k_neuronCount; ++k) {
				w_updateLayerMatrices[l][j][k] = stepFactor
						* dC_dw_layerMatrices[l][j][k];
			}
		}
	}
}

void NetworkTrainer::calculateParameterUpdates(Network &net, double stepFactor,
		double weightDecayFactor) {

	const unsigned int numberOfLayers = net.getNumberOfLayers();

	for (unsigned int l = 1; l < numberOfLayers; ++l) {

		const unsigned int j_neuronCount = net.getNeuronsInLayer()[l];

		for (unsigned int j = 0; j < j_neuronCount; ++j) {

			const unsigned int k_neuronCount = net.getNeuronsInLayer()[l - 1];

			b_updateLayerVectors[l][j] = stepFactor * dC_db_layerVectors[l][j];

			for (unsigned int k = 0; k < k_neuronCount; ++k) {
				w_updateLayerMatrices[l][j][k] = -weightDecayFactor
						* net.getWeightMatrixOfLayer()[l][j][k]
						+ stepFactor * dC_dw_layerMatrices[l][j][k];
			}
		}
	}
}

void NetworkTrainer::calculateParameterUpdates(Network &net, double stepFactor,
		double weightDecayFactor, double frictionFactor) {

	const unsigned int numberOfLayers = net.getNumberOfLayers();

	for (unsigned int l = 1; l < numberOfLayers; ++l) {

		const unsigned int j_neuronCount = net.getNeuronsInLayer()[l];

		for (unsigned int j = 0; j < j_neuronCount; ++j) {

			const unsigned int k_neuronCount = net.getNeuronsInLayer()[l - 1];

			u_layerVectors[l][j] = frictionFactor * u_layerVectors[l][j] + stepFactor * dC_db_layerVectors[l][j];

			b_updateLayerVectors[l][j] = u_layerVectors[l][j];

			for (unsigned int k = 0; k < k_neuronCount; ++k) {

				v_layerMatrices[l][j][k] = frictionFactor * v_layerMatrices[l][j][k] + stepFactor * dC_dw_layerMatrices[l][j][k];

				w_updateLayerMatrices[l][j][k] = -weightDecayFactor * net.getWeightMatrixOfLayer()[l][j][k] + v_layerMatrices[l][j][k];
			}
		}
	}
}

void NetworkTrainer::resetUpdateData(Network &net) {
	errorLayerVectors.clear();
	dC_db_layerVectors.clear();
	dC_dw_layerMatrices.clear();
	b_updateLayerVectors.clear();
	w_updateLayerMatrices.clear();

	u_layerVectors.clear();
	v_layerMatrices.clear();

	for (unsigned int l = 0; l < net.getNumberOfLayers(); ++l) {
		errorLayerVectors.push_back(std::vector<double> { });
		dC_db_layerVectors.push_back(std::vector<double> { });
		dC_dw_layerMatrices.push_back(std::vector<std::vector<double>> { });
		b_updateLayerVectors.push_back(std::vector<double> { });
		w_updateLayerMatrices.push_back(std::vector<std::vector<double>> { });

		u_layerVectors.push_back(std::vector<double> { });
		v_layerMatrices.push_back(std::vector<std::vector<double>> { });

		if (l >= 1) {
			for (unsigned int j = 0; j < net.getNeuronsInLayer()[l]; ++j) {
				errorLayerVectors[l].push_back(0);
				dC_db_layerVectors[l].push_back(0);
				dC_dw_layerMatrices[l].push_back(std::vector<double> { });
				b_updateLayerVectors[l].push_back(0);
				w_updateLayerMatrices[l].push_back(std::vector<double> { });

				u_layerVectors[l].push_back(0);
				v_layerMatrices[l].push_back(std::vector<double> { });

				for (unsigned int k = 0; k < net.getNeuronsInLayer()[l - 1];
						++k) {
					dC_dw_layerMatrices[l][j].push_back(0);
					w_updateLayerMatrices[l][j].push_back(0);

					v_layerMatrices[l][j].push_back(0);
				}
			}
		}
	}
}

void NetworkTrainer::zeroUpdateData(Network &net, unsigned int numberOfLayers) {

	for (unsigned int l = 1; l < numberOfLayers; ++l) {
		const unsigned int neuronsInLayerJ = net.getNeuronsInLayer()[l];

		for (unsigned int j = 0; j < neuronsInLayerJ; ++j) {
			dC_db_layerVectors[l][j] = 0;

			const unsigned int neuronsInLayerK = net.getNeuronsInLayer()[l - 1];

			for (unsigned int k = 0; k < neuronsInLayerK; ++k) {
				dC_dw_layerMatrices[l][j][k] = 0;
			}
		}
	}
}

void NetworkTrainer::zeroNeuronErrors(Network &net,
		unsigned int numberOfLayers) {

	for (unsigned int l = 1; l < numberOfLayers; ++l) {
		const unsigned int neuronsInLayerJ = net.getNeuronsInLayer()[l];

		for (unsigned int j = 0; j < neuronsInLayerJ; ++j) {
			errorLayerVectors[l][j] = 0;
		}
	}
}
