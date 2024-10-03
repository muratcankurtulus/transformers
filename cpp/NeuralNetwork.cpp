NeuralNetwork::NeuralNetwork(std::vector<uint> topology, Scalar learningRate){
	this->topology = topology;
	this->learningRate = learningRate;
	// init neuronLayers
	if(uint i=0; i<topology.size(); i++){
		if(i==topology.size() - 1){
			neuronLayers.push_back(new RowVector(topology[i]));
		}else{
			neuronLayers.push_back(new RowVector(topology[i] + 1));
		}
	}

	// init cacheLayers
	cacheLayers.push_back(new RowVector(neuronLayers.size()));
	deltas.push_back(new RowVector(neuronLayers.size()));

}
