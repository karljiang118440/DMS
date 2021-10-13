#include "benchmark.h"
#include "cpu.h"
#include "datareader.h"
#include "net.h"
#include "gpu.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>
#include <algorithm>



int run_gaze(cv::Mat& roi, cv::Mat& image, ncnn::Net &landmark, int landmark_size_width, int landmark_size_height, float x1, float y1)
{
    int w = roi.cols;
    int h = roi.rows;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(roi.data, ncnn::Mat::PIXEL_BGR,\
                                                 roi.cols, roi.rows, landmark_size_width, landmark_size_height);
    //数据预处理
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1/127.5f, 1/127.5f, 1/127.5f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = landmark.create_extractor();
    ex.set_num_threads(16);
    ex.input("input", in);
    ncnn::Mat out;
    ex.extract("gaze", out);

    ncnn::Mat gaze;




    printf("gaze %f %f\n", out[0], out[1]);

    int length = 150;

    float dx = -length * sin(out[1]);
    float dy = -length * sin(out[0]);

    printf("dx dy %.2f %.2f\n", dx, dy);
    

    cv::arrowedLine(image, cv::Point(50, 50), cv::Point(50+ dx, 50 + dy), cv::Scalar(255, 0, 255), 2, 8, 0, 0.3);



    float sw, sh;
	sw = (float)w/(float)landmark_size_width;
	sh = (float)h/(float)landmark_size_width;


    // 这里需要对模型进行解析处理



    for (int i = 0; i < 64; i++)
    {
        float px,py;
        px = out[i*2]*landmark_size_width*sw+x1;
        py = out[i*2+1]*landmark_size_width*sh+y1;
	    //画实心点
	    cv::circle(image, cv::Point(px, py), 1, cv::Scalar(255,255,255),-1);
    }

    return 0;
}







void run_gaze_test(cv::Mat& roi, cv::Mat& image, float init_x, float init_y)
{
    int w = roi.cols;
    int h = roi.rows;



    int gaze_size_width = 112;
    int gaze_size_height =112;

    
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(roi.data, ncnn::Mat::PIXEL_BGR,\
                                                 roi.cols, roi.rows, gaze_size_width, gaze_size_height);
    //数据预处理
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1/127.5f, 1/127.5f, 1/127.5f};
    in.substract_mean_normalize(mean_vals, norm_vals);



    ncnn::Net gaze_feature;
    gaze_feature.load_param("../models/gazenet_sim-opt-fp16.param");
    gaze_feature.load_model("../models/gazenet_sim-opt-fp16.bin");


    ncnn::Extractor ex = gaze_feature.create_extractor();

    ex.set_num_threads(16);
    ex.input("input", in);
    ncnn::Mat out;
    ex.extract("gaze", out);

    ncnn::Mat gaze;


    int length = 150;

    float dx = -length * sin(out[1]);
    float dy = -length * sin(out[0]);

    printf("dx dy %.2f %.2f\n", dx, dy);
    

    cv::arrowedLine(image, cv::Point(init_x, init_y), cv::Point(init_x+ dx, init_y + dy), cv::Scalar(255, 0, 255), 2, 8, 0, 0.3);


    // return image;


}













int demo(cv::Mat& image, ncnn::Net &detector, int detector_size_width, int detector_size_height, \
         ncnn::Net &landmark, int landmark_size_width, int landmark_size_height)
{

    cv::Mat bgr = image.clone();
    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB,\
                                                 bgr.cols, bgr.rows, detector_size_width, detector_size_height);

    //数据预处理
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = detector.create_extractor();
    ex.set_num_threads(16);
    ex.input("data", in);
    ncnn::Mat out;
    // ex.extract("output", out);

    ex.extract("output", out);

    

    for (int i = 0; i < out.h; i++)
    {
        float x1, y1, x2, y2, score, label;
        float pw,ph,cx,cy;
        const float* values = out.row(i);
        
        x1 = values[2] * img_w;
        y1 = values[3] * img_h;
        x2 = values[4] * img_w;
        y2 = values[5] * img_h;

        pw = x2-x1;
        ph = y2-y1;
        cx = x1+0.5*pw;
        cy = y1+0.5*ph;

        x1 = cx - 0.55*pw;
        y1 = cy - 0.35*ph;
        x2 = cx + 0.55*pw;
        y2 = cy + 0.55*ph;

        score = values[1];
        label = values[0];

        //处理坐标越界问题
        if(x1<0) x1=0;
        if(y1<0) y1=0;
        if(x2<0) x2=0;
        if(y2<0) y2=0;

        if(x1>img_w) x1=img_w;
        if(y1>img_h) y1=img_h;
        if(x2>img_w) x2=img_w;
        if(y2>img_h) y2=img_h;
        
        //限制人脸关键点检测roi图像>66x66
        if( x2-x1 > 66 && y2 -y1 > 66){
            //截取人体ROI
            cv::Mat roi;
            roi = bgr(cv::Rect(x1, y1, x2-x1, y2-y1)).clone();
            run_gaze(roi, image, landmark, landmark_size_width, landmark_size_height, x1, y1);


            cv::rectangle (image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 255, 0), 1, 1, 0);
        }

    }


    return 0;
}
//摄像头测试



int test_cam()
{
    //定义yoloface-500k人脸检测器
    ncnn::Net detector;  

    detector.load_param("../models/yoloface-500k.param");
    detector.load_model("../models/yoloface-500k.bin");

    int detector_size_width  = 320;
    int detector_size_height = 256;



    //定义106关键点预测器
    ncnn::Net landmark;  
    // landmark.load_param("../models/yolo/landmark106.param");
    // landmark.load_model("../models/yolo/landmark106.bin");

    landmark.load_param("../models/gazenet_sim-opt-fp16.param");
    landmark.load_model("../models/gazenet_sim-opt-fp16.bin");


    int gaze_size_width  =  112;
    int gaze_size_height =  112;

    cv::Mat frame;
    cv::VideoCapture cap(0);

    while (true)
    {
        cap >> frame;
        double start = ncnn::get_current_time();
        demo(frame, detector, detector_size_width, detector_size_height, landmark, gaze_size_width, gaze_size_height);
        //demo(frame, detector, 1280, 1080, landmark, landmark_size_width, landmark_size_height);
        double end = ncnn::get_current_time();
        double time = end - start;
        printf("Time:%7.2f \n",time);

        cv::imshow("demo", frame);




        cv::waitKey(1);
    }
    return 0;
}





int gaze_test_cam()
{
    //定义yoloface-500k人脸检测器
    ncnn::Net detector;  

    detector.load_param("../models/yoloface-500k.param");
    detector.load_model("../models/yoloface-500k.bin");

    int detector_size_width  = 320;
    int detector_size_height = 256;



    //定义106关键点预测器
    ncnn::Net landmark;  
    // landmark.load_param("../models/yolo/landmark106.param");
    // landmark.load_model("../models/yolo/landmark106.bin");

    landmark.load_param("../models/gazenet_sim-opt-fp16.param");
    landmark.load_model("../models/gazenet_sim-opt-fp16.bin");


    int gaze_size_width  =  112;
    int gaze_size_height =  112;

    cv::Mat frame;
    cv::VideoCapture cap(0);

    while (true)
    {
        cap >> frame;
        double start = ncnn::get_current_time();
        demo(frame, detector, detector_size_width, detector_size_height, landmark, gaze_size_width, gaze_size_height);
        //demo(frame, detector, 1280, 1080, landmark, landmark_size_width, landmark_size_height);
        double end = ncnn::get_current_time();
        double time = end - start;
        printf("Time:%7.2f \n",time);

        cv::imshow("demo", frame);




        cv::waitKey(1);
    }
    return 0;
}

