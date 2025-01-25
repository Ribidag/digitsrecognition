/*
 * ImageDataLoader.hpp
 *
 *  Created on: 30 juni 2022
 *      Author: ruben
 */

#ifndef IMAGEDATALOADER_HPP_
#define IMAGEDATALOADER_HPP_

#include <vector>
#include <memory>
#include <string>
#include <cstdint>
#include <fstream>

#include "Image.hpp"

class ImageDataLoader {

private:
	static uint32_t readSwapped(std::ifstream &stream);

public:
	static std::vector<std::unique_ptr<Image>> loadData(
			const std::string &imageDataPath, const std::string &labelDataPath);

	static std::vector<std::unique_ptr<Image>> splitOffValidationData(
			std::vector<std::unique_ptr<Image>> &trainingData,
			unsigned int numberOfExamples);

};

#endif /* IMAGEDATALOADER_HPP_ */
