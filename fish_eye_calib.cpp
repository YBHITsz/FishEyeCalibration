#include "stdio.h"
#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{   
    vector<string> files;
    for(int i=0;i<76;i++)
    {
        char str[50];
        sprintf(str, "/home/yangbang/calib_fish_eye/fish_eye_data/%d.jpg", i+1);
        files.push_back(str);
    }
    const int board_w = 11;
    const int board_h = 8;
    const int NPoints = board_w * board_h;//棋盘格内角点总数
    const int boardSize = 45; //mm
    Mat image,grayimage;
    Size ChessBoardSize = cv::Size(board_w, board_h);
    vector<Point2f> tempcorners;

    int flag = 0;
    flag |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
    flag |= cv::fisheye::CALIB_CHECK_COND;
    flag |= cv::fisheye::CALIB_FIX_SKEW;
    //flag |= cv::fisheye::CALIB_USE_INTRINSIC_GUESS;

    vector<Point3f> object;
    for (int j = 0; j < NPoints; j++)
    {
        object.push_back(Point3f((j % board_w) * boardSize, (j / board_w) * boardSize, 0));
    }

    cv::Matx33d intrinsics;//z:相机内参
    cv::Vec4d distortion_coeff;//z:相机畸变系数

    vector<vector<Point3f> > objectv;
    vector<vector<Point2f> > imagev;

    Size corrected_size(1080, 1240);
    Mat mapx, mapy;
    Mat corrected;

    ofstream intrinsicfile("/home/yangbang/calib_fish_eye/intrinsics.txt");
    ofstream disfile("/home/yangbang/calib_fish_eye/dis_coeff.txt");
    int num = 0;
    bool bCalib = false;
    while (num < files.size())
    {
        image = imread(files[num]);
        if (image.empty())
        {
            break;
        }
        imshow("corner_image", image);
        waitKey(10);
        cvtColor(image, grayimage, CV_BGR2GRAY);
        //imshow("gray",grayimage);
        IplImage tempgray = grayimage;
        //find chessboard
        bool findchessboard = cvCheckChessboard(&tempgray, ChessBoardSize);
        //cout<<findchessboard<<endl;
        if (findchessboard)
        {
            bool find_corners_result = findChessboardCorners(grayimage, ChessBoardSize, tempcorners, CALIB_CB_ADAPTIVE_THRESH+CALIB_CB_NORMALIZE_IMAGE+CALIB_CB_FAST_CHECK);
            if (find_corners_result)
            {
                cornerSubPix(grayimage, tempcorners, cvSize(5, 5), cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
                drawChessboardCorners(image, ChessBoardSize, tempcorners, find_corners_result);
                imshow("corner_image", image);
                cvWaitKey(100);
                objectv.push_back(object);
                imagev.push_back(tempcorners);
                cout << "capture " << num << " pictures" << endl;
            }
        }
        tempcorners.clear();
        num++;
    }
    //calib
    vector<Vec3d> rotation_vectors;
    vector<Vec3d> translation_vectors;
    cv::fisheye::calibrate(objectv, imagev, cv::Size(image.cols,image.rows), intrinsics, distortion_coeff, rotation_vectors, translation_vectors, flag, cv::TermCriteria(3, 20, 1e-6)); 
    //undistort,remap method
    cv::fisheye::initUndistortRectifyMap(intrinsics, distortion_coeff, cv::Matx33d::eye(), intrinsics, corrected_size, CV_16SC2, mapx, mapy);
    std::cout<<"intrinsics:"<<endl<<intrinsics<<endl;
    std::cout<<"distortion_coeff:"<<distortion_coeff<<endl;
    for(int i=0; i<3; ++i)
    {
        for(int j=0; j<3; ++j)
        {
            intrinsicfile<<intrinsics(i,j)<<"\t";
        }
        intrinsicfile<<endl;
    }
    for(int i=0; i<4; ++i)
    {
        disfile<<distortion_coeff(i)<<"\t";
    }
    float total_err=0.0;
    for(int i=0;i<objectv.size();i++)
    {
        vector<Point2f> p_img;
        fisheye::projectPoints(objectv[i],p_img,rotation_vectors[i],translation_vectors[i],intrinsics,distortion_coeff);
        vector<Point2f> tempImagePoint=imagev[i];
        Mat tempImagePointMat = Mat(1,tempImagePoint.size(),CV_32FC2);
        Mat image_points2Mat=Mat(1,p_img.size(),CV_32FC2);
        for(size_t j=0;j<tempImagePoint.size();j++)
        {
            image_points2Mat.at<Vec2f>(0,j)=Vec2f(p_img[j].x,p_img[j].y);
            tempImagePointMat.at<Vec2f>(0,j)=Vec2f(tempImagePoint[j].x,tempImagePoint[j].y);
        }
        float err=norm(image_points2Mat,tempImagePointMat,NORM_L2);
        total_err=total_err+err;
        std::cout<<"error of image "<<i<<":"<<err<<endl;
    }
    std::cout<<"avr_error:"<<total_err/objectv.size()<<std::endl;
    intrinsicfile.close();
    disfile.close();

    num = 0;
    while (num < files.size())
    {
        image = imread(files[num++]);

        if (image.empty())
            break;
        remap(image, corrected, mapx, mapy, INTER_LINEAR, BORDER_TRANSPARENT);

        imshow("corner_image", image);
        imshow("corrected", corrected);
        cvWaitKey(200);
    }
    cv::destroyWindow("corner_image");
    cv::destroyWindow("corrected");

    image.release();
    grayimage.release();
    corrected.release();
    mapx.release();
    mapy.release();

    return 0;
}