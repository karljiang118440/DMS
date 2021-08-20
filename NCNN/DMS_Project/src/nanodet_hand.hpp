//
// Created by Xinghao Chen 2020/7/27
//
#include <iostream>
#include <stdio.h>
#include <array>
#include <vector>
#include <opencv2/opencv.hpp>
#include "stdlib.h"
#include <iostream>
#include <array>
#include <vector>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "arcface.h"

using namespace std;
using namespace cv;

// Adjustable Parameters
const bool largest_face_only=true;
const bool record_face=false;
const int distance_threshold = 90;
const float face_thre=0.40;
const float true_thre=0.89;
const int jump=10;
const int input_width = 320;
const int input_height = 240;

/*  add by karl --20210816
1). resize the size 

*/

const int windows_size  =3;

const int output_width = 320 * windows_size;
const int output_height = 240 * windows_size;





// const string project_path="/home/root/ncnn/face/LiveFaceReco";


const string project_path="/media/jcq/Soft/NCNN/workspace/LiveFaceReco_RaspberryPi";




//end

const cv::Size frame_size = Size(output_width,output_height);
const float ratio_x = (float)output_width/ input_width;
const float ratio_y = (float)output_height/ input_height;


int MTCNNDetection();


