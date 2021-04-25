#include "quad.h"

Quad::Quad()
{
    V00.x = 0.0; V00.y = 0.0;
    V01.x = 0.0; V01.y = 0.0;
    V10.x = 0.0; V10.y = 0.0;
    V11.x = 0.0; V11.y = 0.0;
}

Quad::Quad( const Quad &inQuad )
{
    V00.x = inQuad.V00.x; V00.y = inQuad.V00.y;
    V01.x = inQuad.V01.x; V01.y = inQuad.V01.y;
    V10.x = inQuad.V10.x; V10.y = inQuad.V10.y;
    V11.x = inQuad.V11.x; V11.y = inQuad.V11.y;
}

Quad::Quad( const cv::Point2f &inV00, const cv::Point2f &inV01, const cv::Point2f &inV10, const cv::Point2f &inV11 )
{
    V00.x = inV00.x; V00.y = inV00.y;
    V01.x = inV01.x; V01.y = inV01.y;
    V10.x = inV10.x; V10.y = inV10.y;
    V11.x = inV11.x; V11.y = inV11.y;
}

Quad::~Quad()
{

}

void Quad::operator=( const Quad &inQuad )
{
    V00.x = inQuad.V00.x; V00.y = inQuad.V00.y;
    V01.x = inQuad.V01.x; V01.y = inQuad.V01.y;
    V10.x = inQuad.V10.x; V10.y = inQuad.V10.y;
    V11.x = inQuad.V11.x; V11.y = inQuad.V11.y;
}

double Quad::getMinX() const
{
    float minx = min(V00.x, V01.x);
    minx = min(minx, V10.x);
    minx = min(minx, V11.x);

    return 1.0*minx;
}

double Quad::getMaxX() const
{
    float maxX = max(V00.x, V01.x);
    maxX = max(maxX, V10.x);
    maxX = max(maxX, V11.x);

    return 1.0*maxX;
}

double Quad::getMinY() const
{
    float minY = min(V00.y, V01.y);
    minY = min(minY, V10.y);
    minY = min(minY, V11.y);

    return 1.0*minY;
}

double Quad::getMaxY() const
{
    float maxY = max(V00.y, V01.y);
    maxY = max(maxY, V10.y);
    maxY = max(maxY, V11.y);

    return 1.0*maxY;
}

bool Quad::isPointIn( const cv::Point2f &pt ) const
{
    bool in1 = isPointInTriangular(pt, V00, V01, V11);
    bool in2 = isPointInTriangular(pt, V00, V10, V11);
    if (in1 || in2)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool Quad::isPointInTriangular( const cv::Point2f &pt, const cv::Point2f &V0, const cv::Point2f &V1, const cv::Point2f &V2 ) const
{
    double lambda1 = ((V1.y-V2.y)*(pt.x-V2.x) + (V2.x-V1.x)*(pt.y-V2.y)) / ((V1.y-V2.y)*(V0.x-V2.x) + (V2.x-V1.x)*(V0.y-V2.y));
    double lambda2 = ((V2.y-V0.y)*(pt.x-V2.x) + (V0.x-V2.x)*(pt.y-V2.y)) / ((V2.y-V0.y)*(V1.x-V2.x) + (V0.x-V2.x)*(V1.y-V2.y));
    double lambda3 = 1-lambda1-lambda2;
    if (lambda1 >= 0.0 && lambda1 <= 1.0 && lambda2 >= 0.0 && lambda2 <= 1.0 && lambda3 >= 0.0 && lambda3 <= 1.0)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool Quad::getBilinearCoordinates( const cv::Point2f &pt, vector<double> &coefficient ) const
{
    double k1 = -1;
    double k2 = -1;

    double a_x = V00.x - V01.x - V10.x + V11.x;
    double b_x = -V00.x + V01.x;
    double c_x = -V00.x + V10.x;
    double d_x = V00.x - pt.x;

    double a_y = V00.y - V01.y - V10.y + V11.y;
    double b_y = -V00.y + V01.y;
    double c_y = -V00.y + V10.y;
    double d_y = V00.y - pt.y;

    double bigA = -a_y*b_x + b_y*a_x;
    double bigB = -a_y*d_x - c_y*b_x + d_y*a_x +b_y*c_x;
    double bigC = -c_y*d_x + d_y*c_x;

    double tmp1 = -1;
    double tmp2 = -1;
    double tmp3 = -1;
    double tmp4 = -1;

    if (bigB*bigB - 4*bigA*bigC >= 0.0)
    {
        if (abs(bigA) >= 0.000001)
        {
            tmp1 = ( -bigB + sqrt(bigB*bigB - 4*bigA*bigC) ) / ( 2*bigA );
            tmp2 = ( -bigB - sqrt(bigB*bigB - 4*bigA*bigC) ) / ( 2*bigA );
        }
        else
        {
            tmp1 = -bigC/bigB;
        }

        if ( tmp1 >= -0.999999 && tmp1 <= 1.000001)
        {
            tmp3 = -(b_y*tmp1 + d_y) / (a_y*tmp1 + c_y);
            tmp4 = -(b_x*tmp1 + d_x) / (a_x*tmp1 + c_x);
            if (tmp3 >= -0.999999 && tmp3 <= 1.000001)
            {
                k1 = tmp1;
                k2 = tmp3;
            }
            else if (tmp4 >= -0.999999 && tmp4 <= 1.000001)
            {
                k1 = tmp1;
                k2 = tmp4;
            }
        }
        if ( tmp2 >= -0.999999 && tmp2 <= 1.000001)
        {
            tmp3 = -(b_y*tmp2 + d_y) / (a_y*tmp2 + c_y);                        //????????????????
            tmp4 = -(b_x*tmp2 + d_x) / (a_x*tmp2 + c_x);
            if (tmp3 >= -0.999999 && tmp3 <= 1.000001)
            {
                k1 = tmp2;
                k2 = tmp3;
            }
            else if (tmp4 >= -0.999999 && tmp4 <= 1.000001)
            {
                k1 = tmp2;
                k2 = tmp4;
            }
        }
    }
    if (k1>=-0.999999 && k1<=1.000001 && k2>=-0.999999 && k2<=1.000001)
    {
        double coe1 = (1.0-k1)*(1.0-k2);
        double coe2 = k1*(1.0-k2);
        double coe3 = (1.0-k1)*k2;
        double coe4 = k1*k2;

        coefficient.push_back(coe1);
        coefficient.push_back(coe2);
        coefficient.push_back(coe3);
        coefficient.push_back(coe4);

        return true;
    }
    else
    {
        return false;
    }

}

