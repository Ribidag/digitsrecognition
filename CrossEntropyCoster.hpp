/*
 * CrossEntropyCoster.hpp
 *
 *  Created on: 13 juli 2022
 *      Author: ruben
 */

#ifndef CROSSENTROPYCOSTER_HPP_
#define CROSSENTROPYCOSTER_HPP_

#include <vector>
#include <memory>

#include "Image.hpp"
#include "Network.hpp"

class CrossEntropyCoster {

public:
	static double calculateTotalCost(Network &net,
			const std::vector<std::unique_ptr<Image>> &examples);

	static double calculateExampleErrorInNeuron(double a,
			double y);

};

#endif /* CROSSENTROPYCOSTER_HPP_ */
