// SkinDetection.cpp : �������̨Ӧ�ó������ڵ㡣
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
    // ͼƬ����
    Mat image = imread("6.jpg");
    resize(image, image, Size(image.cols  , image.rows ));


    //���ɷ�ɫ��Բģ��
    Mat skinCrCbHist_ellipse = MatCrCbHist_ellipse();

    //�����μ������
    Mat skinCrCbHist_square = MatCrCbHist_square();

    // imshow("������", skinCrCbHist_square);

    Mat detect_ellipse = skin_detect(image, skinCrCbHist_ellipse);
    Mat detect_square = skin_detect(image, skinCrCbHist_square);
    imshow("ԭͼ", image);
    imshow("��ɫ���ͼ-��Բ����", detect_ellipse);
    // imshow("��ɫ���ͼ-��������", detect_square);

    // sobel���ӻ���ݶ�ͼ��
    Mat grad = img_sobel(image);
    imshow("sobel������", grad);

    // ����⵽��ͼ���ֵ��
    Mat bwDetect_RGB;
    threshold(detect_ellipse, bwDetect_RGB, 0.0, 255.0, THRESH_BINARY);
    Mat bwDetect;
    cvtColor(bwDetect_RGB, bwDetect,CV_RGB2GRAY );
    imshow("��ֵ��-����", bwDetect);

    Mat filtered = skin_sobel_filter(bwDetect, grad);
    Mat filtered_detect;
    image.copyTo(filtered_detect, filtered);//���ط�ɫͼ
    imshow("�ۺϿ���", filtered_detect);
    waitKey();
    return 0;
}
/*
    src ������ͼƬ
    skinCrCbHist ��ɫ�������
    ���أ���⵽��ɫ���������
    */
Mat skin_detect(Mat &src, Mat &skinCrCbHist)
{
    Mat img = src.clone();
 

    Mat ycrcb_image;
    Mat output_mask = Mat::zeros(img.size(), CV_8UC1);
    cvtColor(img, ycrcb_image, CV_BGR2YCrCb); //����ת���ɵ�YCrCb�ռ�
    for (int i = 0; i < img.cols; i++)
    {
        for (int j = 0; j < img.rows; j++)
        {
            Vec3b ycrcb = ycrcb_image.at<Vec3b>(j, i);
            if (skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) > 0)//����õ�����Ƥ��ģ����Բ�����ڣ���õ���Ƥ�����ص㡣
                output_mask.at<uchar>(j, i) = 255;
        }
    }
    Mat detect;
    img.copyTo(detect, output_mask);//���ط�ɫͼ
    return detect;
}
/*
    ��sobel�㷨����ÿ��ͼƬ������
*/
Mat img_sobel(Mat &src) {
    Mat src_gray;
    Mat grad;
    int ddepth = CV_16S;
    // Convert the image to grayscale
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Sobel(src_gray, grad_x, ddepth, 1, 0);// x����
    Sobel(src_gray, grad_y, ddepth, 0, 1);// y����
    // converting back to CV_8U
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
    return grad;
}

/*
    ͳ��ÿ�������ڵ������ҳ��������Ƥ��������
*/
Mat skin_sobel_filter(Mat &src, Mat &sobel) {
    vector<vector<cv::Point>> contours;
    Mat filtered = Mat::zeros(src.size(), CV_8UC1);
    // �ҵ�����������Ե��Ϣ�洢��contours
    findContours(src, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_NONE);


    // ����ÿ������
    for (vector<Point> c : contours) {
        Mat mask = Mat(src.size(), CV_8UC1, Scalar(0));
        double area = contourArea(c); // �����������
        double sum = 0; //
        vector<vector<Point>> t_contours;
        if (area <= 10)
            continue;
        // ���ɰ��л��������ͼ��
        t_contours.push_back(c);
        drawContours(mask, t_contours, -1, Scalar(255), CV_FILLED);
        // ���������sobel�ݶȻ���
        for (size_t y = 0; y < src.cols; y++)
        {
            for (size_t x = 0; x < src.rows; x++)
            {
                //Vec3b ycrcb = ycrcb_image.at<Vec3b>(j, i);
                // ����Ƿ���������
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

        cout << "ƽ���ȣ�" << smooth << " �����" << area;
        if (smooth >= 5) {
            cout << " ���/ƽ���ȣ�" << area_smooth << endl;
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
    //����Opencv�Դ�����Բ���ɺ�������һ����ɫ��Բģ��
    Mat skinCrCbHist_ellipse = Mat::zeros(Size(256, 256), CV_8UC1);//256*256�ľ����൱��CrCb�����ĺ�������

    ellipse(skinCrCbHist_ellipse, Point(113, 155.6), Size(23.4, 15.2), 43.0, 0.0, 360.0, Scalar(255, 255, 255), -1);
    return skinCrCbHist_ellipse;
}

Mat MatCrCbHist_square() {
    //�����μ������
    Mat skinCrCbHist_square = Mat::zeros(Size(256, 256), CV_8UC1);//256*256�ľ����൱��CrCb�����ĺ�������
                                                                  //���ɼ���������                                                               
    Point PointArray[4];
    // YCrCb 133<=Cr<=173 77<=Cb<=127
    PointArray[0] = Point(77, 133);
    PointArray[1] = Point(127, 133);
    PointArray[2] = Point(127, 173);
    PointArray[3] = Point(77, 173);
    fillConvexPoly(skinCrCbHist_square, PointArray, 4, Scalar(255, 255, 255));// ��������
    return skinCrCbHist_square;
}