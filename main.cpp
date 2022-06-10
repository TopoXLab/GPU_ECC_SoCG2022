/*
* This codes implements a GPU version ECC computation without futher optimization.
* Author: Fan Wang
* Date: 09/08/2021
*/

#include <iostream>

#include "routines.h"

using namespace std;

int main(int argc, char **argv)
{
	if (argc != 7) helper_();

	bool pad_by_one		 = true;
	bool async_mode		 = true;
	bool mt_read	     = true;
	bool manual_timing   = true;
	bool verbose         = false;
	int imageH_			 = std::atoi(argv[4]);
	int imageW_			 = std::atoi(argv[5]);
	int imageD_			 = std::atoi(argv[6]);
	std::vector<std::string> args;
	args.assign(argv + 1, argv + 4);
	if (args[0] == "b")
		std::vector<double> time = ECC_folder(args[1], imageH_, imageW_, imageD_, pad_by_one, async_mode, mt_read, manual_timing, verbose);
	else if (args[0] == "s")
		std::vector<double> time = ECC_(args[1], imageH_, imageW_, imageD_, pad_by_one, async_mode, mt_read, manual_timing, verbose);
	else helper_();
	

	//// ------ Read/Generate input data
	//std::string fileName0 = "E:/WorkBench/ECC_CPU/example_data/test_float_1000x1000x100.raw";
	//std::string fileName4 = "E:/Data2/SoCG2022/GaussRandomField/3D/3D_512_0.dat";
	//std::string fileName1 = "E:/Data2/SoCG2022/UniformRandom/3D/3D_2048_0.dat";
	//std::string fileName2 = "E:/Data2/SoCG2022/VICTRE/dense_287_359_202_250um_01.dat";
	//std::string fileName3 = "E:/Data2/SoCG2022/CMB/planck13_1500_750.dat";
	//std::string fileName5 = "E:/Data2/SoCG2022/Binary/3D/3D_512_0.dat";
	//std::string fileName6 = "E:/Data2/SoCG2022/GaussRandomField/3D/3D_128_0.dat";
	//std::string fileName7 = "E:/Data2/SoCG2022/Repeated_Gaussian/2D_1024_11.dat";

	//std::string path1 = "E:/Data2/SoCG2022/Stitch/3D";

	//int imageH_        = 2048;
	//int imageW_        = 2048;
	//int imageD_        = 2048;
	//bool pad_by_one    = true;
	//bool async_mode    = true;
	//bool manual_timing = true;
	//bool data_3D       = false;
	//bool mt_read       = true;
	//bool verbose       = false;

	//// ------- Read in data -------
	////float* toy2D = generate_toy_sample(2, 0);
	////float* toy3D = generate_toy_sample(3, 0);
	////float* input_host = from_random_2D_<float>(imageH_, imageW_, 256, 0.2);
	////float* input_host = from_random_3D_<float>(imageH_, imageW_, imageD_, 256, 0.2);
	////write_stream_<float, float>(fileName, input_host, imageH_* imageW_* imageD_);

	//std::vector<double> time = ECC_(fileName1, imageH_, imageW_, imageD_, pad_by_one, async_mode, mt_read, manual_timing, verbose);
	////std::vector<double> time = ECC_folder(path1, imageH_, imageW_, imageD_, pad_by_one, async_mode, mt_read, manual_timing, verbose);
	////std::vector<double> time = ECC_vanila(fileName7, imageH_, imageW_, mt_read, manual_timing, verbose);
	////run_folder(fileName1, 10);
	////run_GENERAL(fileName3, 2, 10);

	return 0;
}