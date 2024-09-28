#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <vector>

Eigen::MatrixXf Matrix;

int main(){
	Matrix << 1.0f, 2.0f, 3.0f,
				    4.0f, 5.0f, 6.0f,
						7.0f, 8.0f, 9.0f;

	std::vector<Matrix*> mtrx;
	std::cout << mtrx << std::endl;
	return 0;
}

