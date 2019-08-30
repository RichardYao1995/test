#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
using namespace std; 

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include "rtabmap/core/util2d.h"
using namespace rtabmap;

inline cv::Point3f project2Dto3D(int x, int y, float d, float fx, float fy,
                                     float cx, float cy, float scale)
{
    float zz = d / scale;
    float xx = zz * (x - cx) / fx;
    float yy = zz * (y - cy) / fy;
    return cv::Point3f(xx, yy, zz);
}

Eigen::Matrix4f pose_estimation_3d3d (const std::vector<cv::Point3f> point1,
                                      const std::vector<cv::Point3f> point2)
{
    cv::Point3f p1, p2;     // center of mass
    int N = point1.size();
    for(int i = 0;i < N;i++)
    {
        p1 += point1[i];
        p2 += point2[i];
    }
    p1 = cv::Point3f(cv::Vec3f(p1) / N);
    p2 = cv::Point3f(cv::Vec3f(p2) / N);
    std::vector<cv::Point3f> q1(N), q2(N); // remove the center
    for(int i = 0;i < N;i++)
    {
        q1[i] = point1[i] - p1;
        q2[i] = point2[i] - p2;
    }

    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for(int i = 0;i < N;i++)
    {
        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    }

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd (W, Eigen::ComputeFullU|Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    if(U.determinant() * V.determinant() < 0)
    {
        for(int x = 0;x < 3;++x)
        {
            U(x, 2) *= -1;
        }
    }

    Eigen::Matrix3d R_ = U * (V.transpose());
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

    Eigen::Matrix4f T;
    T << R_(0,0),R_(0,1),R_(0,2),t_(0,0),
         R_(1,0),R_(1,1),R_(1,2),t_(1,0),
         R_(2,0),R_(2,1),R_(2,2),t_(2,0),
         0, 0, 0, 1;
    return T;
}

int main()
{
    float cx = 640;
    float cy = 320;
    float fx = 762.72;
    float fy = 762.72;
    std::vector< cv::Point2f > keypoints1, keypoints2, kp1, kp2;      // 因为要删除跟踪失败的点，使用list
    cv::Mat color, color1, right, right1, gray, gray1, gray_right, gray_right1;
    color = cv::imread( "/home/uisee/Data/stereo-0/left/0000000814.tiff" );
    color1 = cv::imread( "/home/uisee/Data/stereo-0/left/0000000823.tiff" );
    right = cv::imread( "/home/uisee/Data/stereo-0/right/0000000814.tiff" );
    right1 = cv::imread( "/home/uisee/Data/stereo-0/right/0000000823.tiff" );
    cv::cvtColor(color, gray, CV_BGR2GRAY);
    cv::cvtColor(color1, gray1, CV_BGR2GRAY);
    cv::cvtColor(right, gray_right, CV_BGR2GRAY);
    cv::cvtColor(right1, gray_right1, CV_BGR2GRAY);
    vector<cv::KeyPoint> kps;
    cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
    detector->detect( color, kps );
    for ( auto kp:kps )
        keypoints1.push_back(kp.pt);
    std::vector<unsigned char> status;
    std::vector<float> error;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK( color, color1, keypoints1, keypoints2, status, error );
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;
    for(int i = 0;i < keypoints1.size();i++)
    {
        if(status[i] == 0)
            continue;
        kp1.push_back(keypoints1[i]);
        kp2.push_back(keypoints2[i]);
    }

    cv::Mat disparity = util2d::disparityFromStereoImages(gray, gray_right);
    cv::Mat disparity1 = util2d::disparityFromStereoImages(gray1, gray_right1);
    std::vector<cv::Point3f> vec1, vec2;

    for(int i = 0;i < kp1.size();i++)
    {
        float disp1 = disparity.type() == CV_16SC1 ? float(disparity.at<short>(kp1[i].x, kp1[i].y)) / 16.0f
                                                  :disparity.at<float>(kp1[i].x, kp1[i].y);
        float disp2 = disparity1.type() == CV_16SC1 ? float(disparity1.at<short>(kp2[i].x, kp2[i].y)) / 16.0f
                                                  :disparity1.at<float>(kp2[i].x, kp2[i].y);
        if(disp1 < 3)
            continue;
        if(disp2 < 3)
            continue;
        float d1 = fx * 0.35 / disp1;
        float d2 = fx * 0.35 / disp2;
        cv::Point3f point1 = project2Dto3D(kp1[i].x, kp1[i].y, d1,fx,fy,cx,cy,1.0);
        cv::Point3f point2 = project2Dto3D(kp2[i].x, kp2[i].y, d2,fx,fy,cx,cy,1.0);
        if(abs(point1.z - point2.z) > 2)
            continue;
        vec1.push_back(point1);
        vec2.push_back(point2);
    }
    Eigen::Matrix4f pose = pose_estimation_3d3d(vec1, vec2);
    cout << pose;

    cv::Mat img_show = color.clone();
    for ( auto kp:keypoints1)
        cv::circle(img_show, kp, 10, cv::Scalar(0, 240, 0), 1);
    cv::imshow("corners", img_show);
    cv::waitKey(0);

    return 0;
}
