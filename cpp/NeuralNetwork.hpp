// include the necessary libs for NeuralNetwork processes
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <vector>

// typedefs for easy syntax
typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;

// define the NeuralNetwork class

class NeuralNetwork {
	public:
		// constructor
		NeuralNetwork(std::vector<uint> topology, Scalar learningRate = Scalar(0.005));

		// func for forward propagation
		void propagateForward();

		// func for backward propagation
		void propagateBackward();

		// func for calculating the loss
		void calcLoss();

		// func for updating weights
		void updateWeights();

		// func to train network
		void train();

		std::vector<RowVector*> neuronLayers;
		std::vector<RowVector*> cacheLayers;
		std::vector<RowVector*> deltas;
		std::vector<Matrix*> weights;
		Scalar learningRate;
};
