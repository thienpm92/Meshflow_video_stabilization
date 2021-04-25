#ifndef MESHFLOW_H
#define MESHFLOW_H

#include "mesh.h"
#include "quad.h"
#include <time.h>
#include <fstream>
#include <iostream>

using namespace std;


class MeshFlowVS
{

    public:
         MeshFlowVS(string src_path,string dst_path);
        ~MeshFlowVS();

    private:
         /*fix parameter */
         const int max_corners = 1000;
         const double quality_level = 0.05;
         double minDistance = 5;


        bool fullscreen;
        cv::Mat m_previousFrame;
        cv::Mat m_currentFrame;
        cv::Mat wrp_currentFrame;
        bool m_ProcessFlag;
        string videopath;
        string savevideo;

        vector<vector<cv::Point2f>> localFeatures;
        vector<cv::Point2f> glFeatures;

        vector<vector<cv::Point2f>> localFeatures2;
        vector<cv::Point2f> glFeatures2;


        vector<Quad> vquad;
        vector<cv::Mat> homographV;     //Stores global homography for each frame
        vector<cv::Point2f> originPath;     //It represents C(t) in the paper, which is v(t) and v(t) is the motion vector of each frame.
        vector<cv::Point2f> optimizePath;

        vector<cv::Mat> frameV;     //Store up to 40 video frames
        Mesh* m_mesh;
        Mesh* m_warpedmesh;

        vector<vector<cv::Point2f>> originPathV; //The original path used to store each frame, it needs to be clear that its size should remain the same as FrameV.
        vector<vector<cv::Point2f>> optimizePathV;//An optimized path for storing all the frames in the buffer obtained during one iteration
        vector<vector<vector<cv::Point2f>>> optimizationPathVV;//Store the optimized path for all iterations
        vector<cv::Mat>  subImages;

        int img_width;
        int img_height;
        int subImageWidth;
        int subImageHeight;

        cv::Mat xMat;
        cv::Mat yMat;
        cv::Mat m_globalHomography;

        //Coordinate movement
        vector<cv::Point2f> m_vertexMotion;
        Mesh* m_warpedemesh;

        int m_quadWidth, m_quadHeight;//quad
        int m_meshHeight, m_meshwidth;//mesh


        int m_xGridNum;
        int m_yGridNum;

        int max_iter = 20;
        int iter = 0;
        vector<cv::Mat> subimagt;
        vector<cv::Point2f> update_t;

        vector<Quad> paintQuad;
        vector<Quad> originQuad;



        int m_imagePos;


        string imageSavepath;
        int frameNum;
        string videoname;


        cv::Mat subImageX;
        cv::Mat subImageY;

        cv::Mat subgray;

        cv::Mat xMat2;
        cv::Mat yMat2;


        vector<cv::Mat> homos;

        vector<double> b;

        vector<double> sV;

        int num;
        cv::Mat sum;
        int matSize;

        string savevideopath;
        string videopath1;

        vector<vector<vector<double>>> G_smoothPath;
        vector<vector<double>> G_originPath;
        vector<vector<double>> G_optimizePath;
        vector<vector<double>> G_orgParameter;
        vector<double> store;
        vector<double> G_update_t;
public:
    void startRun();
    void initOriginPath();
    cv::Point2f Trans(cv::Mat H, cv::Point2f pt);
    void ComputeMotionbyFeature(vector<cv::Point2f> &spt, vector<cv::Point2f> &tpt);
    void computeOriginPath();
    cv::Point2f getVertex(int i, int j);
    void DistributeMotion2MeshVertexes_MedianFilter(vector<cv::Point2f> &features, vector<cv::Point2f> &motions, cv::Point2f &globalMotions);
    void SpatialMedianFilter();
    vector<Quad> getQuad();
    void DetectFeatures(cv::Mat m_frame);
    void initMeshFlow(int width, int height);
    void local2glFratures(vector<cv::Point2f> &glFeatures, vector<vector<cv::Point2f> > lcFeatures);
    cv::Point2f TransPtbyH(cv::Mat H, cv::Point2f &pt);
    double calcTv(cv::Mat Ft);
    double calcuFa(cv::Mat Ft);
    double predictLamda(double Tv, double Fa);
    double calcuLamda(cv::Mat Ft);
    double calcuGuassianWeight(int r, int t, int Omegat);
    void initOptPath();
    void setTemporSmotthRange(int &beginframe, int &endframe, int t);
    void getPathT(int t, vector<cv::Point2f> &optedPath);
    void ViewSynthesis(vector<cv::Point2f> optPath, vector<cv::Point2f> Ct);
    void Jacobicompute(ofstream &Video_smooth);
    void Jacobicompute2(int index);
    void addFeatures();

    void image2Video();

    int getSubIndex(cv::Point2f p);

    void CreateShapepreCons(int k);


    void getWeight(cv::Point2f V1, cv::Point2f V2, cv::Point2f V3, double &u, double &v, double &s);

    void computerHomos(cv::Mat& img1, cv::Mat& img2, int& w, int& h, int& index, int num, bool *isCancel);

    void compute(int k, int matSize, int matrows, vector<double> b, cv::Mat sum, Mesh* m1, Mesh *m2);
    void getHomos(Mesh *m);

    cv::Point2f getPoint(int x, int y);

    //第二次中值滤波
    void SpatialMedianFilter2();

    void viewV();

    void meshWarp(const cv::Mat src, cv::Mat &dst, const Mesh &m1, const Mesh &m2);
    void quadWarp(const cv::Mat src, cv::Mat &dst, const Quad &q1, const Quad &q2);

};
#endif // MESHFLOW_H

void QuickSort(vector<float> &arr, int left, int right);
void FastFeature(cv::Mat src, vector<cv::Point2f> &pp1);

