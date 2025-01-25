/*
 * Network.cpp
 *
 *  Created on: 27 juni 2022
 *      Author: ruben
 */

#include <random>
#include <vector>
#include <memory>
#include <fstream>
#include <iostream>
#include <cstdint>

#include "Network.hpp"

Network::Network(const std::vector<unsigned int> &pNeuronsInLayer) :
		m_neuronsInLayer(pNeuronsInLayer), m_numberOfLayers(
				pNeuronsInLayer.size()) {

	// create the network randomly
	prepareDataStorage();
	randomizeParametersNormDist();
}

Network::Network(const std::string &pFilePath) {

	// read the structure of the net
	std::ifstream stream;
	stream.open(pFilePath, std::ios::binary);
	if (!stream) {
		throw std::runtime_error("Could not load weights and biases.");
	}

	uint32_t storedLayers;
	stream.read(reinterpret_cast<char*>(&storedLayers), sizeof storedLayers);
	m_numberOfLayers = storedLayers;

	for (unsigned int l = 0; l < m_numberOfLayers; ++l) {
		uint32_t numberOfNeurons;
		stream.read(reinterpret_cast<char*>(&numberOfNeurons),
				sizeof numberOfNeurons);
		m_neuronsInLayer.push_back(numberOfNeurons);
	}

	// prepare data storage which matches that structure
	prepareDataStorage();

	// read and store the parameters
	for (unsigned int l = 1; l < m_numberOfLayers; ++l) {
		for (unsigned int j = 0; j < m_neuronsInLayer[l]; ++j) {
			double bias;
			stream.read(reinterpret_cast<char*>(&bias), sizeof bias);
			m_biasVectorOfLayer[l][j] = bias;
		}
	}

	for (unsigned int l = 1; l < m_numberOfLayers; ++l) {
		for (unsigned int j = 0; j < m_neuronsInLayer[l]; ++j) {
			for (unsigned int k = 0; k < m_neuronsInLayer[l - 1]; ++k) {
				double weight;
				stream.read(reinterpret_cast<char*>(&weight), sizeof weight);
				m_weightMatrixOfLayer[l][j][k] = weight;
			}
		}
	}
	stream.close();
}

void Network::prepareDataStorage() {
	for (unsigned int l = 0; l < m_numberOfLayers; ++l) {
		m_activationVectorOfLayer.push_back(std::vector<double> { });
		m_activationInputVectorOfLayer.push_back(std::vector<double> { });
		m_biasVectorOfLayer.push_back(std::vector<double> { });
		m_weightMatrixOfLayer.push_back(std::vector<std::vector<double>> { });

		for (unsigned int j = 0; j < m_neuronsInLayer[l]; ++j) {
			m_activationVectorOfLayer[l].push_back(0);

			if (l >= 1) {
				m_activationInputVectorOfLayer[l].push_back(0);
				m_biasVectorOfLayer[l].push_back(0);
				m_weightMatrixOfLayer[l].push_back(std::vector<double> { });

				for (unsigned int k = 0; k < m_neuronsInLayer[l - 1]; ++k) {
					m_weightMatrixOfLayer[l][j].push_back(0);
				}
			}
		}
	}
}

void Network::randomizeParametersNormDist() {
	std::random_device rd { };
	std::mt19937 gen { rd() };
	std::normal_distribution<> d { 0, 1.0 / 28 };

	for (unsigned int l = 1; l < m_numberOfLayers; ++l) {
		for (unsigned int j = 0; j < m_neuronsInLayer[l]; ++j) {
			m_biasVectorOfLayer[l][j] = d(gen);

			for (unsigned int k = 0; k < m_neuronsInLayer[l - 1]; ++k) {
				m_weightMatrixOfLayer[l][j][k] = d(gen);
			}
		}
	}
}

double Network::activationFunction(double z) {
	return 1 / (1 + exp(-z));
}

double Network::dAF_dz(double z) {
	return exp(-z) / ((1 + exp(-z)) * (1 + exp(-z)));
}

const std::vector<double>& Network::feedForward(
		const std::vector<double> &inputActivations) {

	// clear activation inputs
	for (unsigned int l = 1; l < m_numberOfLayers; ++l) {
		for (unsigned int j = 0; j < m_neuronsInLayer[l]; ++j) {
			m_activationInputVectorOfLayer[l][j] = 0;
		}
	}

	// feed forward
	m_activationVectorOfLayer[0] = inputActivations;

	for (unsigned int l = 1; l < m_numberOfLayers; ++l) {
//              std::cout << "Lager: " << l << std::endl;

		for (unsigned int j = 0; j < m_neuronsInLayer[l]; ++j) {
//                      std::cout << " - Neuron: " << j << std::endl;

			for (unsigned int k = 0; k < m_neuronsInLayer[l - 1]; ++k) {
//                              std::cout << "    - Previous neuron (" << k << ") activation: " << activationVectorOfLayer[l - 1][k] << std::endl;

				m_activationInputVectorOfLayer[l][j] +=
						m_weightMatrixOfLayer[l][j][k]
								* m_activationVectorOfLayer[l - 1][k];
			}

			m_activationInputVectorOfLayer[l][j] += m_biasVectorOfLayer[l][j];
			m_activationVectorOfLayer[l][j] = activationFunction(
					m_activationInputVectorOfLayer[l][j]);
//                      std::cout << "       - Aktivering: " << activationVectorOfLayer[l][j] << std::endl;
		}
	}
	return m_activationVectorOfLayer[m_numberOfLayers - 1];
}

void Network::updateWeightsAndBiases(
		const std::vector<std::vector<double> > &biasUpdateLayerVectors,
		const std::vector<std::vector<std::vector<double> > > &weightUpdateLayerMatrices) {

	for (unsigned int l = 1; l < m_numberOfLayers; ++l) {
		for (unsigned int j = 0; j < m_neuronsInLayer[l]; ++j) {
			m_biasVectorOfLayer[l][j] += biasUpdateLayerVectors[l][j];
			for (unsigned int k = 0; k < m_neuronsInLayer[l - 1]; ++k) {
				m_weightMatrixOfLayer[l][j][k] += weightUpdateLayerMatrices[l][j][k];
			}
		}
	}
}

void Network::saveWeightsAndBiases(const std::string &filePath) const {
	std::ofstream fileStream(filePath, std::ios::binary);
	if (!fileStream) {
		throw std::runtime_error(
				"cannot access file to save weigths and biases");
	}

	const uint32_t layersToStore = m_numberOfLayers;
	fileStream.write(reinterpret_cast<const char*>(&layersToStore),
			sizeof layersToStore);

	for (unsigned int l = 0; l < m_numberOfLayers; ++l) {
		const uint32_t numberOfNeurons = m_neuronsInLayer[l];
		fileStream.write(reinterpret_cast<const char*>(&numberOfNeurons),
				sizeof numberOfNeurons);
	}

	for (unsigned int l = 1; l < m_numberOfLayers; ++l) {
		for (unsigned int j = 0; j < m_neuronsInLayer[l]; ++j) {
			double bias = m_biasVectorOfLayer[l][j];
			fileStream.write(reinterpret_cast<const char*>(&bias), sizeof bias);
		}
	}

	for (unsigned int l = 1; l < m_numberOfLayers; ++l) {
		for (unsigned int j = 0; j < m_neuronsInLayer[l]; ++j) {
			for (unsigned int k = 0; k < m_neuronsInLayer[l - 1]; ++k) {
				double weight = m_weightMatrixOfLayer[l][j][k];
				fileStream.write(reinterpret_cast<const char*>(&weight),
						sizeof weight);
			}
		}
	}
	fileStream.close();
}

unsigned int Network::getNumberOfLayers() const {
	return m_numberOfLayers;
}

const std::vector<unsigned int>& Network::getNeuronsInLayer() const {
	return m_neuronsInLayer;
}

const std::vector<std::vector<double> >& Network::getActivationVectorOfLayer() const {
	return m_activationVectorOfLayer;
}

const std::vector<std::vector<double> >& Network::getActivationInputVectorOfLayer() const {
	return m_activationInputVectorOfLayer;
}

const std::vector<std::vector<std::vector<double> > >& Network::getWeightMatrixOfLayer() const {
	return m_weightMatrixOfLayer;
}
