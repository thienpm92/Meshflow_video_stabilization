#include "mesh.h"
#include "quad.h"
Mesh::~Mesh()
{
}

Mesh::Mesh()
{
    imgRows = 0;
    imgCols = 0;
    meshWidth = 0;
    meshHeight = 0;
}

Mesh::Mesh( const Mesh &inMesh )
{
    imgRows = inMesh.imgRows;
    imgCols = inMesh.imgCols;
    meshWidth = inMesh.meshWidth;
    meshHeight = inMesh.meshHeight;
    xMat = cv::Mat::zeros(meshHeight, meshWidth, CV_64FC1);
    yMat = cv::Mat::zeros(meshHeight, meshWidth, CV_64FC1);
    for (int i = 0; i < meshHeight; i ++)
    {
        for (int j = 0; j < meshWidth; j ++)
        {
            setVertex(i, j, inMesh.getVertex(i,j));
        }
    }
}

Mesh::Mesh( int rows, int cols )
{
    imgRows = rows;
    imgCols = cols;
    meshWidth = 0;
    meshHeight = 0;
}

Mesh::Mesh( int rows, int cols, double quadWidth, double quadHeight )
{
    imgRows = rows;
    imgCols = cols;
    buildMesh(quadWidth, quadHeight);
    this->quadWidth = quadWidth;
    this->quadHeight = quadHeight;
}

void Mesh::buildMesh( double quadWidth, double quadHeight )
{
    vector<double> xSet;
    vector<double> ySet;

    for (double x = 0; imgCols - x > 0.5*quadWidth; x += quadWidth)
    {
        xSet.push_back(x);
    }
    xSet.push_back(imgCols-1);
    for (double y = 0; imgRows - y > 0.5*quadHeight; y += quadHeight)
    {
        ySet.push_back(y);
    }
    ySet.push_back(imgRows-1);

    meshWidth = xSet.size();
    meshHeight = ySet.size();

    xMat.create(meshHeight, meshWidth, CV_64FC1);
    yMat.create(meshHeight, meshWidth, CV_64FC1);

    for (int y = 0; y < meshHeight; y ++)
    {
        for (int x = 0; x < meshWidth; x ++)
        {
            xMat.at<double>(y,x) = xSet[x];
            yMat.at<double>(y,x) = ySet[y];
        }
    }
}


void Mesh::setVertex( int i, int j, const cv::Point2f &pos )
{
    xMat.at<double>(i,j) = pos.x;
    yMat.at<double>(i,j) = pos.y;
}

cv::Point2f Mesh::getVertex( int i, int j ) const
{
    double x;
    double y;
    x = xMat.at<double>(i,j);
    y = yMat.at<double>(i,j);
    return cv::Point2f(x,y);
}

void Mesh::updateVertex( int i, int j, const cv::Point2f &pos )
{
    xMat.at<double>(i,j) += pos.x;
    yMat.at<double>(i,j) += pos.y;
}

Quad Mesh::getQuad(int i,int j) const
{
    cv::Point2f V00;
    cv::Point2f V01;
    cv::Point2f V10;
    cv::Point2f V11;

    V00 = getVertex(i-1,j-1);
    V01 = getVertex(i-1,j);
    V10 = getVertex(i,j-1);
    V11 = getVertex(i,j);

    Quad qd(V00,V01,V10,V11);

    return qd;
}

void Mesh::operator=( const Mesh &inMesh )
{
    imgRows = inMesh.imgRows;
    imgCols = inMesh.imgCols;
    meshWidth = inMesh.meshWidth;
    meshHeight = inMesh.meshHeight;
    xMat = cv::Mat::zeros(meshHeight, meshWidth, CV_64FC1);
    yMat = cv::Mat::zeros(meshHeight, meshWidth, CV_64FC1);
    for (int i = 0; i < meshHeight; i ++)
    {
        for (int j = 0; j < meshWidth; j ++)
        {
            setVertex(i, j, inMesh.getVertex(i,j));
        }
    }
}

cv::Mat Mesh::getXMat()
{
    return xMat;
}

cv::Mat Mesh::getYMat()
{
    return yMat;
}

void Mesh::drawMesh( cv::Mat &targetImg){

    cv::Mat temp = targetImg.clone();
    //cv::Scalar color(0,0,0);
    cv::Scalar color(255,255,255);
    int gap = 0;
    int lineWidth=3;

    for (int i = 1; i < meshHeight; i ++)
    {
        for (int j = 1; j < meshWidth; j ++)
        {
            cv::Point2f pUp = getVertex(i-1, j);
            cv::Point2f pLeft = getVertex(i, j-1);
            cv::Point2f pCur = getVertex(i, j);

            pUp.x += gap;
            pUp.y += gap;
            pLeft.x += gap;
            pLeft.y += gap;
            pCur.x += gap;
            pCur.y += gap;

            if(pUp.x > -9999.0 && pUp.y > -9999.0 && pCur.x > -9999.0 && pCur.y > -9999.0){
                double dis = sqrt((pUp.x - pCur.x)*(pUp.x - pCur.x) + (pUp.y - pCur.y)*(pUp.y - pCur.y));
                //if(dis<100){
                    line(temp, cv::Point2f(pUp.x,pUp.y), cv::Point2f(pCur.x,pCur.y),color,lineWidth,CV_AA);
                //}
            }
            if(pLeft.x > -9999.0 && pLeft.y > -9999.0 && pCur.x > -9999.0 && pCur.y > -9999.0){
                double dis = sqrt((pLeft.x - pCur.x)*(pLeft.x - pCur.x) + (pLeft.y - pCur.y)*(pLeft.y - pCur.y));
                //if(dis<100){
                    line(temp, cv::Point2f(pLeft.x,pLeft.y),cv::Point2f(pCur.x,pCur.y), color,lineWidth,CV_AA);
                //}
            }
            cv::circle(temp,cv::Point(pUp.x,pUp.y),lineWidth+2,cv::Scalar(45,57,167),-1);
            cv::circle(temp,cv::Point(pLeft.x,pLeft.y),lineWidth+2,cv::Scalar(45,57,167),-1);
            cv::circle(temp,cv::Point(pCur.x,pCur.y),lineWidth+2,cv::Scalar(45,57,167),-1);
        }
    }

    for (int i = 1; i < meshHeight; i ++)
    {
        cv::Point2f pLeft = getVertex(i, 0);
        cv::Point2f pLeftUp = getVertex(i-1,0);

        pLeftUp.x += gap;
        pLeftUp.y += gap;
        pLeft.x += gap;
        pLeft.y += gap;

        if (pLeft.x > -9999.0 && pLeft.y > -9999.0 && pLeftUp.x > -9999.0 && pLeftUp.y > -9999.0){
            double dis = sqrt((pLeft.x - pLeftUp.x)*(pLeft.x - pLeftUp.x) + (pLeft.y - pLeftUp.y)*(pLeft.y - pLeftUp.y));
            //if(dis<100){
                line(temp, cv::Point2f(pLeft.x,pLeft.y), cv::Point2f(pLeftUp.x,pLeftUp.y),color,lineWidth,CV_AA);
            //}
        }
        cv::circle(temp,cv::Point(pLeftUp.x,pLeftUp.y),lineWidth+2,cv::Scalar(45,57,167),-1);
        cv::circle(temp,cv::Point(pLeft.x,pLeft.y),lineWidth+2,cv::Scalar(45,57,167),-1);
    }

    for (int j = 1; j < meshWidth; j++)
    {
        cv::Point2f pLeftUp = getVertex(0, j-1);
        cv::Point2f pUp = getVertex(0, j);

        pLeftUp.x += gap;
        pLeftUp.y += gap;
        pUp.x += gap;
        pUp.y += gap;

        if (pLeftUp.x > -9999.0 && pLeftUp.y > -9999.0 && pUp.x > -9999.0 && pUp.y > -9999.0){
            double dis = sqrt((pLeftUp.x - pUp.x)*(pLeftUp.x - pUp.x) + (pLeftUp.y - pUp.y)*(pLeftUp.y - pUp.y));
            //if(dis<100){
                line(temp, cv::Point2f(pLeftUp.x,pLeftUp.y), cv::Point2f(pUp.x,pUp.y),color,lineWidth,CV_AA);
            //}
        }
        cv::circle(temp,cv::Point(pUp.x,pUp.y),lineWidth+2,cv::Scalar(45,57,167),-1);
        cv::circle(temp,cv::Point(pLeftUp.x,pLeftUp.y),lineWidth+2,cv::Scalar(45,57,167),-1);
    }
    targetImg = (2.0/5 * targetImg + 3.0/5 *temp);
}





