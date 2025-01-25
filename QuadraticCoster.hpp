/*
 * QuadraticCoster.hpp
 *
 *  Created on: 12 juli 2022
 *      Author: ruben
 */

#ifndef QUADRATICCOSTER_HPP_
#define QUADRATICCOSTER_HPP_

#include <vector>
#include <memory>

#include "Image.hpp"
#include "Network.hpp"

class QuadraticCoster {

public:
	static double calculateTotalCost(Network &net,
			const std::vector<std::unique_ptr<Image>> &examples);

	static double calculateExampleErrorInNeuron(double a,
			double y, double z);

};

#endif /* QUADRATICCOSTER_HPP_ */
