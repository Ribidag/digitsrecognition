/*
 * NetworkTester.cpp
 *
 *  Created on: 7 juli 2022
 *      Author: ruben
 */

#include "NetworkTester.hpp"

#include <vector>
#include <memory>
#include <iostream>

#include <SFML/Graphics.hpp>

#include "Network.hpp"

#include "Image.hpp"

unsigned int NetworkTester::evaluateOutput(
		const std::vector<double> &outputVector) {

	double max = -1;
	unsigned int value = -1;
	for (unsigned int i = 0; i < outputVector.size(); ++i) {
		if (outputVector[i] > max) {
			max = outputVector[i];
			value = i;
		}
	}
	return value;
}

double NetworkTester::testNetwork(Network &net,
		const std::vector<std::unique_ptr<Image>> &testingData) {

	double correct = 0;
	double attempts = testingData.size();
	unsigned int L = net.getNumberOfLayers() - 1;
	for (unsigned int i = 0; i < attempts; ++i) {
		net.feedForward(testingData[i]->getInputActivations());
		auto result = evaluateOutput(net.getActivationVectorOfLayer()[L]);
		auto desiredResult = evaluateOutput(
				testingData[i]->getDesiredOutputActivations());
		if (result == desiredResult) {
			++correct;
		}
	}
	return correct / attempts;
}

void NetworkTester::outputExampleGuessAndAnswer(Network &net,
		std::unique_ptr<Image> &trainingExample) {

	int correctDigit = evaluateOutput(trainingExample->getDesiredOutputActivations());
	int guessedDigit = evaluateOutput(net.feedForward(trainingExample->getInputActivations()));

	std::cout << "Guess: " << guessedDigit << std::endl;
	std::cout << "Answer: " << correctDigit << std::endl;

	if (correctDigit == guessedDigit) {
		std::cout << "Correct!" << std::endl;
	} else {
		std::cout << "Not Correct!" << std::endl;
	}
	std::cout << std::endl;
}

void NetworkTester::updateImageTextureWithGuessAndAnswer(Network &net,
		std::unique_ptr<Image> &testingExample, sf::Texture &texture,
		unsigned int imageSideLength) {

	testingExample->loadPixelsToTexture(texture, imageSideLength);
	outputExampleGuessAndAnswer(net, testingExample);
}
