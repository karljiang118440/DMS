#include "anchor_generator.h"
#include "opencv2/opencv.hpp"
#include "config.h"
#include "tools.h"
#include "pfpld.id.h"


#define _NCNN_PARAM False  // use ncnnoptimize tools to optimize models by karl:20210528



/*

karl:20210528
 
 1):更改模型，使用 ncnnoptimize tools 优化后的模型

 2） 添加 fps 程序，查看优化后的效果进行比较


karl:20210528


*/

/*
1). add image show function 

*/

int pfpld_detect_display(cv::Mat & img){

	// cv::Mat img;


	//  add fps karl:20210528

	// char string[10];
	// double t =0;
	// double fps;
	

	cv::VideoCapture cap(0);

	while(true){

		cap >> img;

		if(img.empty()){

			printf("capture faild \n");
			return -1;
		}
	}

    cv::imshow("img", img);

	// t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

	// fps = 1.0 /t;

	// sprintf(string, "%.2f",fps);

	// std::string fpsString("FPS: ");
	// fpsString += string;

	// printf("%f \n", fps);

	// cv::putText(img,fpsString,cv::Point(100,100),cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(100,0,0));

    // if(cv::waitKey(20) > 0)
    //     break;


}


int pfpld_detect(cv::Mat & img_cnn) {
    extern float pixel_mean[3];
    extern float pixel_std[3];

	#ifdef _NCNN_PARAM // use ncnnoptimize tools to optimize models by karl:20210528

	std::string param_path =  "../models/pfpld/scrfd_500m-opt2.param";
	std::string bin_path = "../models/pfpld/scrfd_500m-opt2.bin";
	std::string pfpld_path = "../models/pfpld/pfpld.ncnnmodel";

	#else 

	std::string param_path =  "../models/pfpld/retina.param";
	std::string bin_path = "../models/pfpld/retina.bin";
	std::string pfpld_path = "../models/pfpld/pfpld.ncnnmodel";

	#endif 


	ncnn::Net _net, pfpld_net;
	_net.load_param(param_path.data());
	_net.load_model(bin_path.data());

	FILE *fp = fopen(pfpld_path.c_str(), "rb");
	if (fp != nullptr) {
		pfpld_net.load_param_bin(fp);
		pfpld_net.load_model(fp);
		fclose(fp);
	}
            

	ncnn::Mat input = ncnn::Mat::from_pixels_resize(img_cnn.data, ncnn::Mat::PIXEL_BGR2RGB, img_cnn.cols, img_cnn.rows, img_cnn.cols, img_cnn.rows);

//    cv::resize(img, img, cv::Size(300, 300));

    input.substract_mean_normalize(pixel_mean, pixel_std);
	ncnn::Extractor _extractor = _net.create_extractor();
	_extractor.input("data", input);


    std::vector<AnchorGenerator> ac(_feat_stride_fpn.size());
    for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
        int stride = _feat_stride_fpn[i];
        ac[i].Init(stride, anchor_cfg[stride], false);
    }

    std::vector<Anchor> proposals;
    proposals.clear();

    for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
    	ncnn::Mat cls;
    	ncnn::Mat reg;
    	ncnn::Mat pts;

        // get blob output
        char clsname[100]; sprintf(clsname, "face_rpn_cls_prob_reshape_stride%d", _feat_stride_fpn[i]);
        char regname[100]; sprintf(regname, "face_rpn_bbox_pred_stride%d", _feat_stride_fpn[i]);
        char ptsname[100]; sprintf(ptsname, "face_rpn_landmark_pred_stride%d", _feat_stride_fpn[i]);
        _extractor.extract(clsname, cls);
        _extractor.extract(regname, reg);
        _extractor.extract(ptsname, pts);

        ac[i].FilterAnchor(cls, reg, pts, proposals);

        for (int r = 0; r < proposals.size(); ++r) {
            proposals[r].print();
        }
    }

    // nms
    std::vector<Anchor> result;
    nms_cpu(proposals, nms_threshold, result);

    // printf("final result %d\n", result.size());
    for(int i = 0; i < result.size(); i ++)
    {
        cv::rectangle (img_cnn, cv::Point((int)result[i].finalbox.x, (int)result[i].finalbox.y), cv::Point((int)result[i].finalbox.width, (int)result[i].finalbox.height), cv::Scalar(255, 255, 0), 2, 8, 0);
//        for (int j = 0; j < result[i].pts.size(); ++j) {
//        	cv::circle(img_cnn, cv::Point((int)result[i].pts[j].x, (int)result[i].pts[j].y), 1, cv::Scalar(225, 0, 225), 2, 8);
//        }
        int x1 = (int)result[i].finalbox.x;
        int y1 = (int)result[i].finalbox.y;
        int x2 = (int)result[i].finalbox.width;
        int y2 = (int)result[i].finalbox.height;
		int height = img_cnn.rows;
		int width = img_cnn.cols;
		int channel = img_cnn.channels();
        int w = x2 - x1 + 1;
        int h = y2 - y1 + 1;

		int size_w = (int)(MAX(w, h)*0.9);
		int size_h = (int)(MAX(w, h)*0.9);
		int cx = x1 + w / 2;
		int cy = y1 + h / 2;
		x1 = cx - size_w / 2;
		x2 = x1 + size_w;
		y1 = cy - (int)(size_h * 0.4);
		y2 = y1 + size_h;
		
		int left = 0;
		int top = 0;
		int bottom = 0;
		int right = 0;
		if(x1 < 0)
            left = -x1;
		if (y1 < 0)
            top = -y1;
		if (x1 >= width)
            right = x2 - width;
		if (y1 >= height)
            bottom = y2 - height;
		
		x1 = MAX(0, x1);
		y1 = MAX(0, y1);
		
		x2 = MIN(width, x2);
		y2 = MIN(height, y2);
		
		cv::Mat face_img = img_cnn(cv::Rect(x1, y1, x2 - x1, y2 - y1));
		cv::copyMakeBorder(face_img, face_img, top, bottom, left, right, cv::BORDER_CONSTANT, 0);
		

		cv::resize(face_img, face_img, cv::Size(112, 112));
		
		ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(
		(unsigned char*)face_img.data,
		ncnn::Mat::PIXEL_BGR2RGB, 112, 112);
		float mean_vals[3] = {0.0, 0.0, 0.0};
        	float norm_vals[3] = {1 / (float)255.0, 1 / (float)255.0, 1 / (float)255.0};
		ncnn_img.substract_mean_normalize(mean_vals, norm_vals);
		
		ncnn::Extractor pfpld_ex = pfpld_net.create_extractor();
		ncnn::Mat pose, landms;
        std::vector<float> angles;
        std::vector<float> landmarks;
		pfpld_ex.input(pfpld_param_id::BLOB_input, ncnn_img);
		pfpld_ex.extract(pfpld_param_id::BLOB_pose, pose);
		pfpld_ex.extract(pfpld_param_id::BLOB_landms, landms);
		for (int j=0; j<pose.w; j++){
            float tmp_angle = pose[j] * 180.0 / CV_PI;
            angles.push_back(tmp_angle);
		}
		
		for (int j=0; j<landms.w / 2; j++)
		{
            float tmp_x = landms[2 * j] * size_w + x1 - left;
            float tmp_y = landms[2 * j + 1] * size_h + y1 -bottom;
            landmarks.push_back(tmp_x);
            landmarks.push_back(tmp_y);
            cv::circle(img_cnn, cv::Point((int)tmp_x, (int)tmp_y), 1, cv::Scalar(0,255,0), 1);
		}
		std::cout<<angles[0]<<"  "<<angles[1]<<"  "<<angles[2]<<std::endl;
		plot_pose_cube(img_cnn, angles[0], angles[1], angles[2], (int)result[i].pts[2].x, (int)result[i].pts[2].y, w / 2);



	/* 在界面中显示角度 -- add by karl:20210830

	1). puttext 函数
	
	
	*/

    putText(img_cnn, "yaw:" + std::to_string(angles[0]), cv::Point(5, 300), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255),2);
    putText(img_cnn, "pitch:" + std::to_string(angles[1]), cv::Point(5, 320), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255),2);
    putText(img_cnn, "roll:" + std::to_string(angles[2]), cv::Point(5, 340), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255),2);



    }

	// cv::imshow("image", img_cnn);
	// // cv::waitKey(1);



}











