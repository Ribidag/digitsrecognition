/*
 * QuadraticCoster.cpp
 *
 *  Created on: 12 juli 2022
 *      Author: ruben
 */

#include "QuadraticCoster.hpp"

#include <vector>
#include <memory>

#include "Image.hpp"
#include "Network.hpp"

double QuadraticCoster::calculateTotalCost(Network &net,
		const std::vector<std::unique_ptr<Image> > &examples) {

	double cost = 0;
	for (auto &trainingExample : examples) {
		auto &outputActivations = net.feedForward(
				trainingExample->getInputActivations());
		auto &desiredOutputActivations =
				trainingExample->getDesiredOutputActivations();

		const unsigned int L = net.getNumberOfLayers() - 1;
		const unsigned int neuronsInLayerL = net.getNeuronsInLayer()[L];

		for (unsigned int j = 0; j < neuronsInLayerL; ++j) {
			cost += (outputActivations[j] - desiredOutputActivations[j])
					* (outputActivations[j] - desiredOutputActivations[j]) / 2;
		}
	}
	return cost;
}

double QuadraticCoster::calculateExampleErrorInNeuron(double a, double y,
		double z) {

	return (a - y) * Network::dAF_dz(z);
}
