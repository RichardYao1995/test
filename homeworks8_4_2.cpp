#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
#include <ctime>
#include <climits>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "/home/uisee/workspace/rtabmap/corelib/include/rtabmap/core/util2d.h"

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/sba/types_six_dof_expmap.h>


using namespace std;
using namespace cv;
using namespace g2o;

struct Measurement
{
    Measurement ( Eigen::Vector3d p, float g ) : pos_world ( p ), grayscale ( g ) {}
    Eigen::Vector3d pos_world;
    float grayscale;
};

inline Eigen::Vector3d project2Dto3D ( int x, int y, int d, float fx, float fy, float cx, float cy, float scale )
{
    float zz = float ( d ) /scale;
    float xx = zz* ( x-cx ) /fx;
    float yy = zz* ( y-cy ) /fy;
    return Eigen::Vector3d ( xx, yy, zz );
}

inline Eigen::Vector2d project3Dto2D ( float x, float y, float z, float fx, float fy, float cx, float cy )
{
    float u = fx*x/z+cx;
    float v = fy*y/z+cy;
    return Eigen::Vector2d ( u,v );
}

class SparseBA : public ceres::SizedCostFunction<1,6>
{
public:
  cv::Mat * gray_;
  double cx_, cy_;
  double fx_, fy_;

  double pixelValue_;
  double X_, Y_, Z_;

SparseBA(cv::Mat *gray, double cx, double cy, double fx, double fy, double X, double Y, double Z, double pixelValue)
{
  gray_ = gray;
  cx_ = cx;
  cy_ = cy;
  fx_ = fx;
  fy_ = fy;
  X_ = X;
  Y_ = Y;
  Z_ = Z;
  pixelValue_ = pixelValue;
}

virtual bool Evaluate (double const *const *pose, double *residual, double **jacobians) const{
  //存储p的坐标
  double p[3];
  p[0] = X_;
  p[1] = Y_;
  p[2] = Z_;

  //存储新的p'的坐标
  double newP[3];
  double R[3];
  R[0] = pose[0][0];
  R[1] = pose[0][1];
  R[2] = pose[0][2];
  ceres::AngleAxisRotatePoint(R, p, newP);

  newP[0] += pose[0][3];
  newP[1] += pose[0][4];
  newP[2] += pose[0][5];

  //新的p‘点投影到像素坐标系
  double ux = fx_ * newP[0] / newP[2] + cx_;
  double uy = fy_ * newP[1] / newP[2] + cy_;

  residual[0] = getPixelValue(ux, uy) - pixelValue_;

  if (jacobians)
  {
    double invz = 1.0 / newP[2];
    double invz_2 = invz * invz;

    //公式8.15
    Eigen::Matrix<double, 2, 6> jacobian_uv_ksai;
    jacobian_uv_ksai(0,0) = -newP[0] * newP[1] * invz_2 * fx_;
    jacobian_uv_ksai(0,1) = (1 + (newP[0] * newP[0] * invz_2)) * fx_;
    jacobian_uv_ksai(0,2) = -newP[1] * invz * fx_;
    jacobian_uv_ksai(0,3) = invz * fx_;
    jacobian_uv_ksai(0,4) = 0;
    jacobian_uv_ksai(0,5) = -newP[0] * invz_2 * fx_;

    jacobian_uv_ksai(1,0) = -(1 + newP[1] * newP[1] * invz_2) * fy_;
    jacobian_uv_ksai(1,1) = newP[0] * newP[1] * invz_2 * fy_;
    jacobian_uv_ksai(1,2) = newP[0] * invz * fy_;
    jacobian_uv_ksai(1,3) = 0;
    jacobian_uv_ksai(1,4) = invz * fy_;
    jacobian_uv_ksai(1,5) = -newP[1] * invz_2 * fy_;

    //像素梯度
    Eigen::Matrix<double, 1, 2> jacobian_pixel_uv;
    jacobian_pixel_uv(0,0) = (getPixelValue(ux+1, uy) - getPixelValue(ux-1, uy))/2;
    jacobian_pixel_uv(0,1) = (getPixelValue(ux, uy+1) - getPixelValue(ux, uy-1))/2;

    //公式8.16
    Eigen::Matrix<double, 1, 6> jacobian = jacobian_pixel_uv * jacobian_uv_ksai;

    jacobians[0][0] = jacobian(0);
    jacobians[0][1] = jacobian(1);
    jacobians[0][2] = jacobian(2);
    jacobians[0][3] = jacobian(3);
    jacobians[0][4] = jacobian(4);
    jacobians[0][5] = jacobian(5);
  }

  return true;

}

double getPixelValue (double x, double y) const
{
  uchar* data = & gray_->data[int(y) * gray_->step + int(x)];
  double xx = x - floor(x);
  double yy = y - floor(y);
  return double (
    (1 - xx) * (1 - yy) * data[0] + xx * (1 - yy) * data[1] + (1 - xx) * yy * data[gray_->step] + xx * yy * data[gray_->step + 1]
  );
}
};

class EdgeSE3ProjectDirect: public g2o::BaseUnaryEdge< 1, double, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ProjectDirect()
    {}

    EdgeSE3ProjectDirect(Eigen::Vector3d point, float fx, float fy, float cx, float cy, cv::Mat* image)
        :x_world_(point), fx_(fx), fy_(fy), cx_(cx), cy_(cy), image_(image)
    {}

    virtual void computeError()
    {
        const g2o::VertexSE3Expmap* v  = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        Eigen::Vector3d x_local = v->estimate().map(x_world_);
        float x = x_local[0] * fx_ / x_local[2] + cx_;
        float y = x_local[1] * fy_ / x_local[2] + cy_;
        // check x,y is in the image
        if(x - 4 < 0 || (x + 4) > image_->cols || (y - 4) < 0 || (y + 4) > image_->rows)
        {
            _error(0, 0) = 0.0;
            this->setLevel(1);
        }
        else
        {
            _error(0, 0) = getPixelValue(x, y) - _measurement;
        }
    }

    // plus in manifold
    virtual void linearizeOplus()
    {
        if(level() == 1)
        {
            _jacobianOplusXi = Eigen::Matrix<double, 1, 6>::Zero();
            return;
        }
        g2o::VertexSE3Expmap* vtx = static_cast<g2o::VertexSE3Expmap*>(_vertices[0]);
        Eigen::Vector3d xyz_trans = vtx->estimate().map(x_world_);   // q in book

        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double invz = 1.0 / xyz_trans[2];
        double invz_2 = invz * invz;

        float u = x * fx_ * invz + cx_;
        float v = y * fy_ * invz + cy_;

        // jacobian from se3 to u,v
        // NOTE that in g2o the Lie algebra is (\omega, \epsilon),
        //where \omega is so(3) and \epsilon the translation
        Eigen::Matrix<double, 2, 6> jacobian_uv_ksai;

        jacobian_uv_ksai(0, 0) = -x * y * invz_2 * fx_;
        jacobian_uv_ksai(0, 1) = (1 + (x * x * invz_2)) * fx_;
        jacobian_uv_ksai(0, 2) = - y * invz * fx_;
        jacobian_uv_ksai(0, 3) = invz * fx_;
        jacobian_uv_ksai(0, 4) = 0;
        jacobian_uv_ksai(0, 5) = -x * invz_2 * fx_;

        jacobian_uv_ksai(1, 0) = -(1 + y * y * invz_2) * fy_;
        jacobian_uv_ksai(1, 1) = x * y * invz_2 * fy_;
        jacobian_uv_ksai(1, 2) = x * invz * fy_;
        jacobian_uv_ksai(1, 3) = 0;
        jacobian_uv_ksai(1, 4) = invz * fy_;
        jacobian_uv_ksai(1, 5) = -y * invz_2 * fy_;

        Eigen::Matrix<double, 1, 2> jacobian_pixel_uv;

        jacobian_pixel_uv(0, 0) = (getPixelValue(u + 1, v) - getPixelValue(u - 1, v)) / 2;
        jacobian_pixel_uv(0, 1) = (getPixelValue(u, v + 1) - getPixelValue(u, v - 1)) / 2;

        _jacobianOplusXi = jacobian_pixel_uv * jacobian_uv_ksai;
    }

    // dummy read and write functions because we don't care...
    virtual bool read (std::istream& in)
    {}
    virtual bool write (std::ostream& out) const
    {}

protected:
    // get a gray scale value from reference image (bilinear interpolated)
    inline float getPixelValue(float x, float y)
    {
        uchar* data = &image_->data[int(y) * image_->step + int(x)];
        float xx = x - floor(x);
        float yy = y - floor(y);
        return float(
                   (1 - xx) * (1 - yy) * data[0] +
                   xx * (1 - yy) * data[1] +
                   (1 - xx) * yy * data[image_->step] +
                   xx * yy * data[image_->step + 1]);
    }
public:
    Eigen::Vector3d x_world_;   // 3D point in world frame
    float cx_ = 0, cy_ = 0, fx_ = 0, fy_ = 0; // Camera intrinsics
    cv::Mat* image_ = nullptr;    // reference image
};

cv::Point3f projectDisparityTo3D(
        const cv::Point2f & pt,
        float disparity)
{
    if(disparity > 0.0f)
    {
        float W = 0.35/disparity;
        return cv::Point3f((pt.x - 640)*W, (pt.y - 360)*W, 762.72*W);
    }
    float bad_point = std::numeric_limits<float>::quiet_NaN ();
    return cv::Point3f(bad_point, bad_point, bad_point);
}

bool poseEstimationDirect(const std::vector< Measurement >& measurements, cv::Mat* gray, Eigen::Matrix3f& K, Eigen::Isometry3d& Tcw);

int main ()
{       
    cv::Mat left, disparity, right, second, color1, color2;
    color1 = cv::imread("/home/uisee/Data/stereo-0/left/0000000106.tiff");
    color2 = cv::imread("/home/uisee/Data/stereo-0/left/0000000107.tiff");
    right = cv::imread("/home/uisee/Data/stereo-0/right/0000000106.tiff");
    cv::cvtColor(color1, left, CV_BGR2GRAY);
    cv::cvtColor(right, right, CV_BGR2GRAY);
    cv::cvtColor(color2, second, CV_BGR2GRAY);
    disparity = rtabmap::util2d::disparityFromStereoImages(left, right);
    cv::imwrite("disparity.png", disparity);
    //disparity = getDisparity(gray, gray1);

    vector<Measurement> measurements;
    // 相机内参
    float cx = 640;
    float cy = 320;
    float fx = 762.72;
    float fy = 762.72;
    Eigen::Matrix3f K;
    K<<fx,0.f,cx,0.f,fy,cy,0.f,0.f,1.0f;

    Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity();

    // 我们以第一个图像为参考，对后续图像和参考图像做直接法
    for ( int x=10; x<left.cols-10; x++ )
        for ( int y=10; y<left.rows-10; y++ )
        {
            Eigen::Vector2d delta (
                left.ptr<uchar>(y)[x+1] - left.ptr<uchar>(y)[x-1],
                left.ptr<uchar>(y+1)[x] - left.ptr<uchar>(y-1)[x]
            );
            if ( delta.norm() < 80 )
                continue;
            float disp = disparity.type() == CV_16SC1 ? float(disparity.at<short>(y, x)) / 16.0f
                                                      :disparity.at<float>(y, x);
            if(disp < 3)
                continue;

            float d = fx * 0.35 / disp;
            Eigen::Vector3d p3d = project2Dto3D(x, y, d, fx, fy, cx, cy, 1.0);
//            cv::Point3f point = projectDisparityTo3D(cv::Point2f(x, y), disp);
//            Eigen::Vector3d p3d((double)point.x, (double)point.y, (double)point.z);
            float grayscale = float ( left.ptr<uchar> (y) [x] );
            measurements.push_back ( Measurement ( p3d, grayscale ) );
        }
    cout<<"add total "<<measurements.size()<<" measurements."<<endl;

   


        // 使用直接法计算相机运动
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        poseEstimationDirect ( measurements, &second, K, Tcw );
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
        cout<<"direct method costs time: "<<time_used.count() <<" seconds."<<endl;
        cout<<"Tcw="<<Tcw.matrix() <<endl;

        // plot the feature points
        cv::Mat img_show ( color1.rows*2, color1.cols, CV_8UC3 );
        color1.copyTo ( img_show ( cv::Rect ( 0,0,color1.cols, color1.rows ) ) );
        color2.copyTo ( img_show ( cv::Rect ( 0,color1.rows,color1.cols, color1.rows ) ) );
        for ( Measurement m:measurements )
        {
            if ( rand() > RAND_MAX/5 )
                continue;
            Eigen::Vector3d p = m.pos_world;
            Eigen::Vector2d pixel_prev = project3Dto2D ( p ( 0,0 ), p ( 1,0 ), p ( 2,0 ), fx, fy, cx, cy );
            Eigen::Vector3d p2 = Tcw*m.pos_world;
            Eigen::Vector2d pixel_now = project3Dto2D ( p2 ( 0,0 ), p2 ( 1,0 ), p2 ( 2,0 ), fx, fy, cx, cy );
            if ( pixel_now(0,0)<0 || pixel_now(0,0)>=left.cols || pixel_now(1,0)<0 || pixel_now(1,0)>=left.rows )
                continue;

            float b = 0;
            float g = 250;
            float r = 0;
            img_show.ptr<uchar>( pixel_prev(1,0) )[int(pixel_prev(0,0))*3] = b;
            img_show.ptr<uchar>( pixel_prev(1,0) )[int(pixel_prev(0,0))*3+1] = g;
            img_show.ptr<uchar>( pixel_prev(1,0) )[int(pixel_prev(0,0))*3+2] = r;

            img_show.ptr<uchar>( pixel_now(1,0)+left.rows )[int(pixel_now(0,0))*3] = b;
            img_show.ptr<uchar>( pixel_now(1,0)+left.rows )[int(pixel_now(0,0))*3+1] = g;
            img_show.ptr<uchar>( pixel_now(1,0)+left.rows )[int(pixel_now(0,0))*3+2] = r;
            cv::circle ( img_show, cv::Point2d ( pixel_prev ( 0,0 ), pixel_prev ( 1,0 ) ), 4, cv::Scalar ( b,g,r ), 2 );
            cv::circle ( img_show, cv::Point2d ( pixel_now ( 0,0 ), pixel_now ( 1,0 ) +left.rows ), 4, cv::Scalar ( b,g,r ), 2 );
        }
        cv::imwrite("result.png", img_show);
        cv::imshow ( "result", img_show );
        cv::waitKey ( 0 );


    return 0;
}

bool poseEstimationDirect ( const vector< Measurement >& measurements, cv::Mat* gray, Eigen::Matrix3f& K, Eigen::Isometry3d& Tcw )
{
    // 初始化g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,1>> DirectBlock;  // 求解的向量是6＊1的
    DirectBlock::LinearSolverType* linearSolver = new g2o::LinearSolverDense< DirectBlock::PoseMatrixType > ();
    DirectBlock* solver_ptr = new DirectBlock ( linearSolver );
    // g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr ); // G-N
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr ); // L-M
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );
    optimizer.setVerbose( true );

    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setEstimate ( g2o::SE3Quat ( Tcw.rotation(), Tcw.translation() ) );
    pose->setId ( 0 );
    optimizer.addVertex ( pose );

    // 添加边
    int id=1;
    for ( Measurement m: measurements )
    {
        EdgeSE3ProjectDirect* edge = new EdgeSE3ProjectDirect (
            m.pos_world,
            K ( 0,0 ), K ( 1,1 ), K ( 0,2 ), K ( 1,2 ), gray
        );
        edge->setVertex ( 0, pose );
        edge->setMeasurement ( m.grayscale );
        edge->setInformation ( Eigen::Matrix<double,1,1>::Identity() );
        edge->setId ( id++ );
        optimizer.addEdge ( edge );
    }
    cout<<"edges in graph: "<<optimizer.edges().size() <<endl;
    optimizer.initializeOptimization();
    optimizer.optimize(30);
    Tcw = pose->estimate();
}

//bool poseEstimationDirect(const std::vector< Measurement >& measurements, cv::Mat* gray, Eigen::Matrix3f& K, Eigen::Isometry3d& Tcw)
//{
//  ceres::Problem problem;
//  //定义位姿数组
//  double pose[6];
//  //用轴角进行优化
//  Eigen::AngleAxisd rotationVector(Tcw.rotation());
//  pose[0] = rotationVector.angle() * rotationVector.axis()(0);
//  pose[1] = rotationVector.angle() * rotationVector.axis()(1);
//  pose[2] = rotationVector.angle() * rotationVector.axis()(2);
//  pose[3] = Tcw.translation()(0);
//  pose[4] = Tcw.translation()(1);
//  pose[5] = Tcw.translation()(2);

//  //构建Ceres问题
//  for (Measurement m:measurements)
//  {
//    ceres::CostFunction * costFunction = new SparseBA(gray, K(0,2), K(1,2), K(0,0), K(1,1), m.pos_world(0), m.pos_world(1), m.pos_world(2), double(m.grayscale));
//    problem.AddResidualBlock(costFunction, NULL, pose);
//  }

//  ceres::Solver::Options options;
//  options.num_threads = 4;
//  options.linear_solver_type = ceres::DENSE_QR;
//  options.minimizer_progress_to_stdout = true;
//  ceres::Solver::Summary summary;
//  ceres::Solve(options, &problem, &summary);
//  std::cout << 1 << std::endl;

//  cv::Mat rotateVectorCV = cv::Mat::zeros(3, 1, CV_64FC1);
//  rotateVectorCV.at<double>(0) = pose[0];
//  rotateVectorCV.at<double>(1) = pose[1];
//  rotateVectorCV.at<double>(2) = pose[2];

//  cv::Mat RCV;
//  cv::Rodrigues(rotateVectorCV, RCV);
//  Tcw(0,0) = RCV.at<double>(0,0); Tcw(0,1) = RCV.at<double>(0,1); Tcw(0,2) = RCV.at<double>(0,2);
//  Tcw(1,0) = RCV.at<double>(1,0); Tcw(1,1) = RCV.at<double>(1,1); Tcw(1,2) = RCV.at<double>(1,2);
//  Tcw(2,0) = RCV.at<double>(2,0); Tcw(2,1) = RCV.at<double>(2,1); Tcw(2,2) = RCV.at<double>(2,2);

//  Tcw(0,3) = pose[3];
//  Tcw(1,3) = pose[4];
//  Tcw(2,3) = pose[5];
//}

