// SkinDetection.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"


#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <iostream>

using namespace cv;
using namespace std;
Mat img_sobel(Mat &src);
Mat skin_sobel_filter(Mat &src, Mat &sobel);
Mat skin_detect(Mat &src, Mat &skinCrCbHist);

Mat MatCrCbHist_ellipse();
Mat MatCrCbHist_square();

struct skinAreaInfo
{
    Mat mask;
    double area;
    double smooth;
};

int main()
{
    // 图片载入
    Mat image = imread("6.jpg");
    resize(image, image, Size(image.cols  , image.rows ));


    //生成肤色椭圆模型
    Mat skinCrCbHist_ellipse = MatCrCbHist_ellipse();

    //正方形检测区域
    Mat skinCrCbHist_square = MatCrCbHist_square();

    // imshow("检测矩阵", skinCrCbHist_square);

    Mat detect_ellipse = skin_detect(image, skinCrCbHist_ellipse);
    Mat detect_square = skin_detect(image, skinCrCbHist_square);
    imshow("原图", image);
    imshow("肤色检测图-椭圆区域", detect_ellipse);
    // imshow("肤色检测图-矩形区域", detect_square);

    // sobel算子获得梯度图像
    Mat grad = img_sobel(image);
    imshow("sobel运算结果", grad);

    // 将检测到的图像二值化
    Mat bwDetect_RGB;
    threshold(detect_ellipse, bwDetect_RGB, 0.0, 255.0, THRESH_BINARY);
    Mat bwDetect;
    cvtColor(bwDetect_RGB, bwDetect,CV_RGB2GRAY );
    imshow("二值化-人脸", bwDetect);

    Mat filtered = skin_sobel_filter(bwDetect, grad);
    Mat filtered_detect;
    image.copyTo(filtered_detect, filtered);//返回肤色图
    imshow("综合考虑", filtered_detect);
    waitKey();
    return 0;
}
/*
    src 待检测的图片
    skinCrCbHist 肤色区域矩阵
    返回，检测到肤色的区域矩阵
    */
Mat skin_detect(Mat &src, Mat &skinCrCbHist)
{
    Mat img = src.clone();
 

    Mat ycrcb_image;
    Mat output_mask = Mat::zeros(img.size(), CV_8UC1);
    cvtColor(img, ycrcb_image, CV_BGR2YCrCb); //首先转换成到YCrCb空间
    for (int i = 0; i < img.cols; i++)
    {
        for (int j = 0; j < img.rows; j++)
        {
            Vec3b ycrcb = ycrcb_image.at<Vec3b>(j, i);
            if (skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) > 0)//如果该点落在皮肤模型椭圆区域内，则该点是皮肤像素点。
                output_mask.at<uchar>(j, i) = 255;
        }
    }
    Mat detect;
    img.copyTo(detect, output_mask);//返回肤色图
    return detect;
}
/*
    用sobel算法计算每个图片的纹理
*/
Mat img_sobel(Mat &src) {
    Mat src_gray;
    Mat grad;
    int ddepth = CV_16S;
    // Convert the image to grayscale
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Sobel(src_gray, grad_x, ddepth, 1, 0);// x方向
    Sobel(src_gray, grad_y, ddepth, 0, 1);// y方向
    // converting back to CV_8U
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
    return grad;
}

/*
    统计每个区域内的纹理，找出最可能是皮肤的区域
*/
Mat skin_sobel_filter(Mat &src, Mat &sobel) {
    vector<vector<cv::Point>> contours;
    Mat filtered = Mat::zeros(src.size(), CV_8UC1);
    // 找到轮廓，将边缘信息存储至contours
    findContours(src, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_NONE);


    // 遍历每个轮廓
    for (vector<Point> c : contours) {
        Mat mask = Mat(src.size(), CV_8UC1, Scalar(0));
        double area = contourArea(c); // 计算轮廓面积
        double sum = 0; //
        vector<vector<Point>> t_contours;
        if (area <= 10)
            continue;
        // 在蒙版中画出该填充图形
        t_contours.push_back(c);
        drawContours(mask, t_contours, -1, Scalar(255), CV_FILLED);
        // 求出区域内sobel梯度积分
        for (size_t y = 0; y < src.cols; y++)
        {
            for (size_t x = 0; x < src.rows; x++)
            {
                //Vec3b ycrcb = ycrcb_image.at<Vec3b>(j, i);
                // 检查是否在区域内
                /*if (pointPolygonTest(c, Point(x, y), false) <= 0)
                    continue;*/
                if (mask.at<uchar>(x, y) != 255)
                    continue;
                uchar grad = sobel.at<uchar>(x, y);
                sum += grad;
            }
        }
        double smooth = sum / area;
        double area_smooth = area / smooth;

        cout << "平滑度：" << smooth << " 面积：" << area;
        if (smooth >= 5) {
            cout << " 面积/平滑度：" << area_smooth << endl;
        }
        else cout << endl;
        if (area_smooth >= 10)
        {
            drawContours(filtered, t_contours, -1, Scalar(255), CV_FILLED);

        }

    }
    return filtered;
}

Mat MatCrCbHist_ellipse() {
    //利用Opencv自带的椭圆生成函数生成一个肤色椭圆模型
    Mat skinCrCbHist_ellipse = Mat::zeros(Size(256, 256), CV_8UC1);//256*256的矩阵，相当于CrCb分量的横纵坐标

    ellipse(skinCrCbHist_ellipse, Point(113, 155.6), Size(23.4, 15.2), 43.0, 0.0, 360.0, Scalar(255, 255, 255), -1);
    return skinCrCbHist_ellipse;
}

Mat MatCrCbHist_square() {
    //正方形检测区域
    Mat skinCrCbHist_square = Mat::zeros(Size(256, 256), CV_8UC1);//256*256的矩阵，相当于CrCb分量的横纵坐标
                                                                  //生成检测区域矩形                                                               
    Point PointArray[4];
    // YCrCb 133<=Cr<=173 77<=Cb<=127
    PointArray[0] = Point(77, 133);
    PointArray[1] = Point(127, 133);
    PointArray[2] = Point(127, 173);
    PointArray[3] = Point(77, 173);
    fillConvexPoly(skinCrCbHist_square, PointArray, 4, Scalar(255, 255, 255));// 画出矩形
    return skinCrCbHist_square;
}