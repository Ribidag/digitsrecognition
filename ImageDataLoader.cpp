/*
 * ImageDataLoader.cpp
 *
 *  Created on: 30 juni 2022
 *      Author: ruben
 */

#include "ImageDataLoader.hpp"

#include <vector>
#include <memory>
#include <string>
#include <cstdint>
#include <fstream>
#include <iostream>

#include "Image.hpp"

uint32_t ImageDataLoader::readSwapped(std::ifstream &stream) {
	uint32_t a;
	for (unsigned int i = 0; i < sizeof a; ++i) {
		stream.read(reinterpret_cast<char*>(&a) + (sizeof a) - 1 - i, 1);
	}

	return a;
}

std::vector<std::unique_ptr<Image>> ImageDataLoader::loadData(const std::string& imageDataPath, const std::string& labelDataPath) {
	std::ifstream imageDataStream;
	imageDataStream.open(imageDataPath, std::ios::binary);
	if (!imageDataStream) {
		throw std::runtime_error("Could not load images file");
	}

	readSwapped(imageDataStream);
	uint32_t numberOfImages = readSwapped(imageDataStream);
	uint32_t rows = readSwapped(imageDataStream);
	uint32_t columns = readSwapped(imageDataStream);

	std::ifstream labelDataStream;
	labelDataStream.open(labelDataPath, std::ios::binary);
	if (!labelDataStream) {
		throw std::runtime_error("Could not load labels file");
	}

	readSwapped(labelDataStream);
	readSwapped(labelDataStream);

	std::vector<std::unique_ptr<Image>> data;

	for (unsigned int i = 0; i < numberOfImages; ++i) {

		std::vector<double> exInputActivations;
		int exLabel;

		for (unsigned int r = 0; r < rows; ++r) {
			for (unsigned int c = 0; c < columns; ++c) {
				uint8_t pixelValue;
				imageDataStream.read(reinterpret_cast<char*>(&pixelValue), 1);
				exInputActivations.push_back(
						1.0 - static_cast<double>(pixelValue / 256.0));
			}
		}

		uint8_t label;
		labelDataStream.read(reinterpret_cast<char*>(&label), 1);
		exLabel = static_cast<int>(label);
		//if (i == 299) {
			//std::cout << exLabel << std::endl;
		//}

		std::vector<double> exDesiredOutputActivations;
		for (unsigned int i = 0; i < 10; ++i) {
			exDesiredOutputActivations.push_back(0);
		}
		exDesiredOutputActivations[exLabel] = 1;

		data.push_back(std::make_unique<Image>(exInputActivations, exDesiredOutputActivations));
	}
	return data;
}

std::vector<std::unique_ptr<Image> > ImageDataLoader::splitOffValidationData(
		std::vector<std::unique_ptr<Image> > &trainingData,
		unsigned int numberOfExamples) {

	std::vector<std::unique_ptr<Image>> validationData;

	for (unsigned int e = 0; e < numberOfExamples; ++e) {
		validationData.push_back(std::move(trainingData.back()));
		trainingData.pop_back();
	}

	return validationData;
}
