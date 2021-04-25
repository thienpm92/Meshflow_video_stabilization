#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <time.h>
#include "quad.h"
using namespace std;

#ifndef __MESH__
#define __MESH__
class Mesh
{
private:
    cv::Mat xMat;
    cv::Mat yMat;

public:
    int imgRows;
    int imgCols;

    int meshWidth;
    int meshHeight;

    int quadWidth;
    int quadHeight;

    Mesh();
    Mesh(const Mesh &inMesh);
    Mesh(int rows, int cols);
    Mesh(int rows, int cols, double quadWidth, double quadHeight);
    ~Mesh();
    void operator=(const Mesh &inMesh);
    void buildMesh(double quadWidth, double quadHeight);
    void setVertex(int i, int j, const cv::Point2f &pos);
    void updateVertex(int i, int j, const cv::Point2f &pos);
    cv::Point2f getVertex(int i, int j) const;
    Quad getQuad(int i,int j) const;
    void drawMesh( cv::Mat &targetImg);

    cv::Mat getXMat();
    cv::Mat getYMat();
};

#endif


