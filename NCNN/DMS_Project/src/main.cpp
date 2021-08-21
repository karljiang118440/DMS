//
// Created by Xinghao Chen 2020/7/27
//
#include "livefacereco.hpp"
#include "nanodet_hand.hpp"

#include "detect_camera.hpp"




#include <stdio.h>
#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

int main() {
//MTCNN detector(prefix);

// MTCNNDetection();

nanodet_hand();

pfpld_detect();

// runlandmark();


}
