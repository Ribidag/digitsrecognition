/*
 * NetworkTester.hpp
 *
 *  Created on: 7 juli 2022
 *      Author: ruben
 */

#ifndef NETWORKTESTER_HPP_
#define NETWORKTESTER_HPP_

#include <vector>
#include <memory>

#include "Image.hpp"
#include "Network.hpp"

class NetworkTester {

private:
	static void outputExampleGuessAndAnswer(Network &net,
			std::unique_ptr<Image> &trainingExample);

public:
	static unsigned int evaluateOutput(const std::vector<double> &outputVector);

	static double testNetwork(Network &net,
			const std::vector<std::unique_ptr<Image>> &testingExamples);

	static void updateImageTextureWithGuessAndAnswer(Network &net,
			std::unique_ptr<Image> &trainingExample,
			sf::Texture &texture, unsigned int imageSideLength);
};

#endif /* NETWORKTESTER_HPP_ */
