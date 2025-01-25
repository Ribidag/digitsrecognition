/*
 * Image.cpp
 *
 *  Created on: 28 juni 2022
 *      Author: ruben
 */

#include <vector>
#include <cmath>

#include <SFML/Graphics.hpp>
#include "Image.hpp"


Image::Image(const std::vector<double> &pInputActivations,
		const std::vector<double> &pDesiredOutputActivations) :
		inputActivations(pInputActivations), desiredOutputActivations(
				pDesiredOutputActivations) {
}

const std::vector<double>& Image::getInputActivations() const {
	return inputActivations;
}

const std::vector<double>& Image::getDesiredOutputActivations() const {
	return desiredOutputActivations;
}

const std::vector<sf::Uint8> Image::getExPixels() const {

	std::vector<sf::Uint8> pixels;

	for (const double pixelValue : inputActivations) {
		for (unsigned int i = 0; i < 3; ++i) {
			pixels.push_back(std::round(pixelValue * 255));
		}
		pixels.push_back(255);
	}
	return pixels;
}

void Image::loadPixelsToTexture(sf::Texture &texture, const unsigned int imageSideLength) const {
	auto &pixels = getExPixels();

	sf::Image testingImage;
	testingImage.create(imageSideLength, imageSideLength,
			&pixels[0]);
	texture.loadFromImage(testingImage);
}
