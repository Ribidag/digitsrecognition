/*
 * Image.hpp
 *
 *  Created on: 28 juni 2022
 *      Author: ruben
 */

#ifndef IMAGE_HPP_
#define IMAGE_HPP_

#include <vector>

#include <SFML/Graphics.hpp>

class Image {
private:
	const std::vector<double> inputActivations;
	const std::vector<double> desiredOutputActivations;

	const std::vector<sf::Uint8> getExPixels() const;

public:
	Image(const std::vector<double>& pInputActivations, const std::vector<double>& pDesiredOutputActivations);

	const std::vector<double>& getInputActivations() const;
	const std::vector<double>& getDesiredOutputActivations() const;

	void loadPixelsToTexture(sf::Texture &texture, unsigned int imageSideLength) const;
};



#endif /* IMAGE_HPP_ */
