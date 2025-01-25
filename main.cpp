/*
 * main.cpp
 *
 *  Created on: 29 juni 2022
 *      Author: ruben
 */

#include <vector>
#include <memory>
#include <iostream>
#include <string>
#include <iomanip>

#include <SFML/Graphics.hpp>

#include "Image.hpp"
#include "ImageDataLoader.hpp"
#include "NetworkTrainer.hpp"
#include "NetworkTester.hpp"
#include "Network.hpp"

int main() {
	std::cout << std::setprecision(15);

	auto trainingData = ImageDataLoader::loadData(
			"training_and_testing/train-images.idx3-ubyte",
			"training_and_testing/train-labels.idx1-ubyte");

	auto validationData = ImageDataLoader::splitOffValidationData(trainingData,
			10000);

	auto testingData = ImageDataLoader::loadData(
			"training_and_testing/t10k-images.idx3-ubyte",
			"training_and_testing/t10k-labels.idx1-ubyte");

	NetworkTrainer networkTrainer;

//	Network net(std::vector<unsigned int> { 784, 30, 10 });
//	//learning rate, weight decay factor, frictionFactor, mini batch size, number of epochs
//	networkTrainer.stochasticGradientDescent(net, trainingData, 0.001,
//			0.00000005, 0.65, 1, 100, validationData);
//	net.saveWeightsAndBiases("saved_networks/2.txt");

	Network net("saved_networks/1.txt");

	std::cout << std::endl << "correct (testing): " << NetworkTester::testNetwork(net, testingData)
			<< std::endl << std::endl;

	unsigned int windowSideLength = 700;
	unsigned int imageSideLength = 28;
	sf::RenderWindow window(sf::VideoMode(windowSideLength, windowSideLength),
			"Digit Recogniton");
	window.setView(
			sf::View(sf::FloatRect(0, 0, imageSideLength, imageSideLength)));

	sf::Image testingImage;
	testingImage.create(imageSideLength, imageSideLength, sf::Color::Black);
	sf::Texture texture;
	texture.loadFromImage(testingImage);
	sf::Sprite sprite;
	sprite.setTexture(texture, true);

	unsigned int exampleIndex = 2000;
	NetworkTester::updateImageTextureWithGuessAndAnswer(net,
			testingData[exampleIndex], texture, imageSideLength);

	while (window.isOpen()) {
		sf::Event event;
		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed) {
				window.close();
			}
			if (event.type == sf::Event::KeyPressed) {
				if (event.key.code == sf::Keyboard::N) {
					++exampleIndex;
					NetworkTester::updateImageTextureWithGuessAndAnswer(net,
							testingData[exampleIndex], texture,
							imageSideLength);
				}
				if (event.key.code == sf::Keyboard::P) {
					--exampleIndex;
					NetworkTester::updateImageTextureWithGuessAndAnswer(net,
							testingData[exampleIndex], texture,
							imageSideLength);
				}
			}
		}

		window.clear();
		window.draw(sprite);
		window.display();
	}

	return 0;
}
