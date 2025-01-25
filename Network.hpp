/*
 * Network.hpp
 *
 *  Created on: 27 juni 2022
 *      Author: ruben
 */

#ifndef NETWORK_HPP_
#define NETWORK_HPP_

#include <vector>
#include <memory>

class Network {

private:
	std::vector<unsigned int> m_neuronsInLayer;
	unsigned int m_numberOfLayers;

	std::vector<std::vector<double>> m_activationVectorOfLayer;
	std::vector<std::vector<double>> m_activationInputVectorOfLayer;
	std::vector<std::vector<double>> m_biasVectorOfLayer;
	std::vector<std::vector<std::vector<double>>> m_weightMatrixOfLayer;

	void prepareDataStorage();
	void randomizeParametersNormDist();

public:
	Network(const std::vector<unsigned int> &pNeuronsInLayer);
	Network(const std::string &pFilePath);

	static double activationFunction(double z);
	static double dAF_dz(double z);

	const std::vector<double>& feedForward(
			const std::vector<double> &inputActivations);

	void updateWeightsAndBiases(
			const std::vector<std::vector<double>> &biasUpdateLayerVectors,
			const std::vector<std::vector<std::vector<double>>> &weightUpdateLayerMatrices);
	void saveWeightsAndBiases(const std::string &filePath) const;

	unsigned int getNumberOfLayers() const;
	const std::vector<unsigned int>& getNeuronsInLayer() const;
	const std::vector<std::vector<double>>& getActivationVectorOfLayer() const;
	const std::vector<std::vector<double>>& getActivationInputVectorOfLayer() const;
	const std::vector<std::vector<std::vector<double>>>& getWeightMatrixOfLayer() const;

};

#endif /* NETWORK_HPP_ */
