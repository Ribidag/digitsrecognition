/*
 * CrossEntropyCoster.cpp
 *
 *  Created on: 13 juli 2022
 *      Author: ruben
 */

#include "CrossEntropyCoster.hpp"

#include <vector>
#include <memory>
#include <cmath>

#include "Image.hpp"
#include "Network.hpp"

double CrossEntropyCoster::calculateTotalCost(Network &net,
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
			auto a = outputActivations[j];
			auto y = desiredOutputActivations[j];
			cost -= y * std::log(a) + (1 - y) * std::log(1 - a);
		}
	}
	return cost;
}

double CrossEntropyCoster::calculateExampleErrorInNeuron(double a, double y) {

	return a - y;
}
