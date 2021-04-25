#include "meshflow.h"
#include "mesh.h"
#include <string>
#include <stdio.h>

#include <omp.h>
#include <time.h>


#define meshcount 16
#define phi 40
#define mesh 16
#define subImageCount 4
#define Radius 50
#define neighborsnum 20
#define iternum 20

using namespace std;

//The parameter is the width and height of the picture
void MeshFlowVS::initMeshFlow(int width, int height)
{
    img_height = height;
    img_width = width;
    //Divided into 4*4 grids
    m_quadWidth = 1.0*img_width / meshcount;
    m_quadHeight = 1.0*img_height / meshcount;

    m_mesh = new Mesh(img_height, img_width, 1.0*m_quadWidth, 1.0*m_quadHeight);


    m_meshHeight = m_mesh->meshHeight;//The number of grids per column
    m_meshwidth = m_mesh->meshWidth;//The number of grids per row

    Mesh mmesh(img_height, img_width, m_quadWidth, m_quadHeight);//初始化并构建网格
    xMat = mmesh.getXMat();
    yMat = mmesh.getYMat();

    m_vertexMotion.resize(m_meshHeight*m_meshwidth);    //describe motion of vertices (size = all number of vertices)
}


void MeshFlowVS::initOriginPath()
{//The original path is summed by the motion vectors of the grid coordinates
    for (int i = 0; i <mesh + 1; i++)
    {
        for (int j = 0; j < mesh + 1; j++)
        {
            cv::Point2f p(0, 0);
            originPath.push_back(p);
        }
    }
    originPathV.push_back(originPath);
}

//Calculate feature points and multiply the homography matrix
cv::Point2f MeshFlowVS::Trans(cv::Mat H, cv::Point2f pt){
    cv::Point2f result;

    double a = H.at<double>(0, 0) * pt.x + H.at<double>(0, 1) * pt.y + H.at<double>(0, 2);
    double b = H.at<double>(1, 0) * pt.x + H.at<double>(1, 1) * pt.y + H.at<double>(1, 2);
    double c = H.at<double>(2, 0) * pt.x + H.at<double>(2, 1) * pt.y + H.at<double>(2, 2);

    result.x = a / c;
    result.y = b / c;

    return result;
}

//This function can only be used normally from the second frame,
//using the median filter method to spread the motion of the feature point to the coordinate point, equivalent to the f1 in the paper.
void MeshFlowVS::DistributeMotion2MeshVertexes_MedianFilter(vector<cv::Point2f> &features, vector<cv::Point2f> &motions,cv::Point2f &globalMotions)
{
    vector<vector<float>> motionx, motiony;
    motionx.resize(m_meshHeight*m_meshwidth);
    motiony.resize(m_meshHeight*m_meshwidth);


    for (int i = 0; i < m_meshHeight; i++) {
        for (int j = 0; j < m_meshwidth; j++)  {
            //Get the coordinates of the coordinates
            cv::Point2f pt = m_mesh->getVertex(i, j);
            //Finding the feature points around this coordinate is not necessarily correct,
            //but it may also be based on the coordinates of the feature points to find the coordinates of the grid
            for (int k = 0; k < features.size(); k++)
            {
                cv::Point2f pt2 = features[k];
                //Calculate the distance between the two
                float dis = sqrt((pt.x - pt2.x)*(pt.x - pt2.x) + (pt.y - pt2.y)*(pt.y - pt2.y));
                float disx = abs(pt.x - pt2.x);
                float disy = abs(pt.y - pt2.y);
                if (disx <= m_quadWidth*2 && disy<=m_quadHeight*2)
                {
                    motionx[i*m_meshwidth + j].push_back(motions[k].x);
                    motiony[i*m_meshwidth + j].push_back(motions[k].y);
                }
            }
        }
    }
    //Median selection
  //  #pragma omp parallel for
    for (int i = 0; i < m_meshHeight; i++)
    {
        //#pragma omp parallel for
        for (int j = 0; j <m_meshwidth; j++)
        {
            if (motionx[i*m_meshwidth + j].size()>1)
            {
                //Quickly sort movements in x and y directions
                QuickSort(motionx[i*m_meshwidth + j], 0, motionx[i*m_meshwidth + j].size() - 1);
                QuickSort(motiony[i*m_meshwidth + j], 0, motiony[i*m_meshwidth + j].size() - 1);

                //Take median value
                m_vertexMotion[i*m_meshwidth + j].x = motionx[i*m_meshwidth + j][motionx[i*m_meshwidth + j].size() / 2];
                m_vertexMotion[i*m_meshwidth + j].y = motiony[i*m_meshwidth + j][motiony[i*m_meshwidth + j].size() / 2];
            }
            else{
                m_vertexMotion[i*m_meshwidth + j].x = globalMotions.x;
                m_vertexMotion[i*m_meshwidth + j].y = globalMotions.y;
            }
        }
    }
}


//Second median filter
void MeshFlowVS::SpatialMedianFilter2()
{
    //Since the number of grids is 16*16, the coordinates are 17*17
    vector<cv::Point2f> tempVertexMotion(m_meshHeight*m_meshwidth);
    for (int i = 0; i < m_meshHeight; i++)
    {
        for (int j = 0; j <m_meshwidth; j++)
        {
            tempVertexMotion[i*m_meshwidth + j] = m_vertexMotion[i*m_meshwidth + j];
        }
    }

    int radius = 10;

    for (int i = 0; i <= m_meshHeight-radius; i++)
    {
      //  #pragma omp parallel for
        for (int j = 0; j <= m_meshwidth-radius; j++)
        {
            vector<float> motionx;
            vector<float> motiony;

            for (int k = 0; k < radius; k++)
            {
                int m = (i+k) * 17 + j;

                for (int l = 0; l < radius; l++)
                {

                    int t = m + l;
                    if (t < tempVertexMotion.size())
                    {
                        motionx.push_back(tempVertexMotion[t].x);
                        motiony.push_back(tempVertexMotion[t].y);
                    }
                }
            }
            QuickSort(motionx, 0, motionx.size() - 1);
            QuickSort(motiony, 0, motiony.size() - 1);
            if (j == m_meshwidth - radius)
            {
                for (int tt = 0; tt < radius; tt++)
                {
                    //Here is just for this line
                    m_vertexMotion[i*m_meshwidth + j+tt].x = motionx[motionx.size() / 2];
                    m_vertexMotion[i*m_meshwidth + j+tt].y = motiony[motiony.size() / 2];
                }
            }
            else
            {
                m_vertexMotion[i*m_meshwidth + j].x = motionx[motionx.size() / 2];
                m_vertexMotion[i*m_meshwidth + j].y = motiony[motiony.size() / 2];
            }
        }

        if (i == m_meshHeight - radius)
        {
            for (int j = 0; j <= m_meshwidth - radius; j++)
            {
                vector<float> motionx;
                vector<float> motiony;
                for (int k = 0; k < radius; k++)
                {
                    int m = (i + k) * 17 + j;
                    for (int l = 0; l < radius; l++)
                    {
                        int t = m + l;
                        if (t < tempVertexMotion.size())
                        {
                            motionx.push_back(tempVertexMotion[t].x);
                            motiony.push_back(tempVertexMotion[t].y);
                        }
                    }
                }

                QuickSort(motionx, 0, motionx.size() - 1);
                QuickSort(motiony, 0, motiony.size() - 1);


                if (j == m_meshwidth - radius)
                {
                    for (int p = 0; p < radius; p++)
                    {
                        for (int tt = 0; tt <radius ; tt++)
                        {
                            m_vertexMotion[(i + p)*m_meshwidth + j+tt].x = motionx[motionx.size() / 2];
                            m_vertexMotion[(i + p)*m_meshwidth + j+tt].y = motiony[motiony.size() / 2];
                        }
                    }
                }
                else
                {
                    for (int tt = 0; tt < radius; tt++)
                    {
                        m_vertexMotion[(i+tt)*m_meshwidth + j].x = motionx[motionx.size() / 2];
                        m_vertexMotion[(i+tt)*m_meshwidth + j].y = motiony[motiony.size() / 2];

                    }
                }
            }
        }
    }
}


//Here, the number of grid coordinates is 17*17, which may be different in the first median filter.
void MeshFlowVS::computeOriginPath()
{
    for (int i = 0; i <m_meshHeight; i++)
    {
        for (int j = 0; j < m_meshwidth; j++)
        {
            m_warpedemesh->updateVertex(i,j,m_vertexMotion[i*m_meshwidth + j]);
        }
    }
    for (int i = 0; i < originPath.size(); i++)
    {
        originPath[i].x += m_vertexMotion[i].x;
        originPath[i].y += m_vertexMotion[i].y;
    }
    originPathV.push_back(originPath);

    m_vertexMotion.clear();
}


//Synthetic video frames
void MeshFlowVS::ViewSynthesis(vector<cv::Point2f> optPath, vector<cv::Point2f> Ct)
{
    if(frameNum==1){
        update_t.clear();
        update_t.resize(Ct.size());
        for (int i = 0; i < Ct.size(); i++)  {
            update_t[i] = optPath[i] - Ct[i];
        }
    }

    for (int i = 0; i <m_meshHeight; i++)   {
        for (int j = 0; j < m_meshwidth; j++)  {
            m_warpedemesh->updateVertex(i,j,update_t[i*m_meshHeight+j]);
        }
    }
}

//Set the start time and end time of the smooth time according to different situations
void MeshFlowVS::setTemporSmotthRange(int &beginframe, int &endframe, int t)
{
    const int k = 10 / 2;
    int range;
    //Find the distance from frame t to both ends
    int ds = t;
    int de = frameV.size() - 1 - t;

    range = ds < de ? ds : de;
    range = range < k ? range : k;
    beginframe = t - range;
    endframe = t + range;
}

//Calculate lamda
double MeshFlowVS::calcuLamda(cv::Mat Ft)
{
    double	Fa = calcuFa(Ft);
    double  Tv = calcTv(Ft);
    return predictLamda(Tv, Fa);
}

//Calculate Fa
double MeshFlowVS::calcuFa(cv::Mat Ft)
{
    //First fill in 3*3
    Ft.at<double>(2, 0) = 0;
    Ft.at<double>(2, 1) = 0;
    Ft.at<double>(2, 2) = 1;

    //Finding characteristic values
    cv::Mat eValuesMat;

    eigen(Ft, eValuesMat);
    vector<double> a;
    for (int i = 0; i < eValuesMat.rows; i++)
    {
        for (int j = 0; j < eValuesMat.cols; j++)
        {
            a.push_back(eValuesMat.at<double>(i, j));
        }
    }

    //Find the two maximums
    double max1, max2;
    //Sort first
    sort(a.begin(), a.begin() + 3);

    max1 = a[1];
    max2 = a[2];
    return max1 / max2;
}

//Calculate Tv, whose parameter Ft is the global homography
double MeshFlowVS::calcTv(cv::Mat Ft)
{
    double vx = Ft.at<double>(0, 2);
    double vy = Ft.at<double>(1, 2);
    return sqrt(vx*vx + vy*vy);
}

//predict Lamda
double MeshFlowVS::predictLamda(double Tv, double Fa)
{
    double lamda1 = -1.93*Tv + 0.95;
    double lamda2 = 5.83*Fa + 4.88;

    double result = lamda1 < lamda2 ? lamda1 : lamda2;

    return result>0 ? result : 0;
}

//The first is to calculate the first t-1 frames, Omega represents the time smoothing radius, r is one of the values, t is a frame number
double MeshFlowVS::calcuGuassianWeight(int r, int t, int Omegat)
{
    return exp(-pow((r - t), 2.0) / (pow(Omegat / 3.0, 2.0)));
}

void MeshFlowVS::Jacobicompute(ofstream &Video_smooth)
{
    if (iter == 0)
    {//The third frame is processed at this time, and the first iteration is because this iteration involves the last optimization result.
        vector<vector<cv::Point2f>> optiPath;
        vector<cv::Point2f> p1 = originPathV[0];
        vector<cv::Point2f> p2 = originPathV[1];
        optiPath.push_back(p1);
        optiPath.push_back(p2);
        //Take the original path of the first and second frames as the result of the first optimization
        optimizationPathVV[max_iter-1] = optiPath;

    }

    //The optimized path will be saved to optimizePathV
    optimizePathV.clear();
    optimizePathV.resize(frameV.size());
    G_optimizePath.clear();
    G_optimizePath.resize(G_originPath.size());
    for (int pri = 0; pri <frameV.size(); pri++)
    {
        optimizePathV[pri] = originPathV[pri];
    }
    for(int i=0; i<max_iter;i++){
        optimizationPathVV[i] = originPathV;
        G_smoothPath[i] = (G_originPath);
    }
    for (int it = 1; it < max_iter; it++)
    {//Iteration 20 times
        for (int t = 0; t < frameV.size(); t++)
        {
            //Due to the Gaussian distribution, the optimization starts from the first frame and ends at the penultimate frame.
            //The last frame is optimized by other methods.
            int beginframe = 0, endframe = frameV.size() - 1;

            //Get the optimized time radius of the optimized frame
//            setTemporSmotthRange(beginframe, endframe, t);

            //The number of homography matrices should be one less than the number of video frames
 //           double lamda = calcuLamda(homographV[t - 1]);
            double lamda = 100;
            //Get the original path of the frame to be optimized
            vector<cv::Point2f> Ct = originPathV[t];

            //Get the optimized path of the frame at the previous iteration
            vector<cv::Point2f> prePt;
            if(t==0){
                prePt = optimizationPathVV[optimizationPathVV.size()-1][t];
            }
            else{
                prePt = optimizationPathVV[optimizationPathVV.size()-1][t-1];
            }
            double sum_wtr = 0;
            double gamme = 0;
            //Time smoothing radius
            int omegat = (endframe - beginframe + 1)/2;
            //initialization
            vector<cv::Point2f> sum_wtrP(Ct.size(), cv::Point2f(0, 0));

            for (int r = beginframe; r <= endframe; r++)
            {
                if (r != t)
                {//Obtain                   
                    vector<cv::Point2f> Pr = optimizationPathVV[it - 1][r];
                    //The first iteration sets the initial value, calculates and saves the weight
                    double tempWeight = calcuGuassianWeight(r, t, omegat);
                    sum_wtr += tempWeight;

                    for (int pri = 0; pri < Pr.size(); pri++)
                    {
                        Pr[pri].x *= tempWeight;
                        Pr[pri].y *= tempWeight;
                        sum_wtrP[pri].x += Pr[pri].x;
                        sum_wtrP[pri].y += Pr[pri].y;
                    }
                }
            }

            gamme = 1 + lamda*sum_wtr;
            double wc = 1, wpre = 1, we = lamda;
            wc = wc / gamme;
            wpre = wpre / gamme;
            we = we / gamme;

            //Ct by wc
            for (int i = 0; i < Ct.size(); i++)
            {
                Ct[i].x *= wc;
                Ct[i].y *= wc;
            }

            //Prept by wpre
            for (int i = 0; i < prePt.size(); i++)
            {
                prePt[i].x *= wpre;
                prePt[i].y *= wpre;
            }

            //We multiply by sum_wtrP
            for (int wi = 0; wi < sum_wtrP.size(); wi++)
            {
                sum_wtrP[wi].x *= we;
                sum_wtrP[wi].y *= we;
            }

            //Add three parts
            for (int pi = 0; pi < Ct.size(); pi++)
            {
                optimizationPathVV[it][t][pi].x = Ct[pi].x + prePt[pi].x + sum_wtrP[pi].x;
                optimizationPathVV[it][t][pi].y = Ct[pi].y + prePt[pi].y + sum_wtrP[pi].y;
            }     
        }


        for (int t = 0; t < G_originPath.size(); t++)
        {
            int beginF = 0, endF = G_originPath.size() - 1;
            vector<double> G_Ct = G_originPath[t];
            vector<double> G_prePt;
            if(t==0){
                G_prePt = G_smoothPath[G_smoothPath.size() - 1][t];
            }
            else
                G_prePt = G_smoothPath[G_smoothPath.size() - 1][t-1];
            double G_sum_wtr = 0;
            double G_gamme = 0;
            int G_omegat = (endF - beginF + 1)/2;
            double lamda = 50;
            vector<double> G_sum_wtrP(4,0);
            for (int r = beginF; r <= endF; r++){
                if (r != t)
                {
                    vector<double> G_Pr = G_smoothPath[it-1][r];
                    vector<double> G_Cr = G_originPath[r];

                    double gaussWeight = calcuGuassianWeight(r, t, G_omegat);
                    G_sum_wtr += (gaussWeight);
                    G_Pr[0] *= (gaussWeight);
                    G_Pr[1] *= (gaussWeight);
                    G_Pr[2] *= (gaussWeight);
                    G_Pr[3] *= (gaussWeight);

                    G_sum_wtrP[0] += G_Pr[0];
                    G_sum_wtrP[1] += G_Pr[1];
                    G_sum_wtrP[2] += G_Pr[2];
                    G_sum_wtrP[3] += G_Pr[3];
                }
            }
            G_gamme = 1 + lamda*G_sum_wtr;
            double G_wc = 1.0, G_wpre = 1.0, G_we = lamda;
            G_wc = G_wc / G_gamme;
            G_wpre = G_wpre / G_gamme;
            G_we = G_we / G_gamme;
            G_Ct[0] *= G_wc;
            G_Ct[1] *= G_wc;
            G_Ct[2] *= G_wc;
            G_Ct[3] *= G_wc;

            G_prePt[0] *= G_wpre;
            G_prePt[1] *= G_wpre;
            G_prePt[2] *= G_wpre;
            G_prePt[3] *= G_wpre;

            G_sum_wtrP[0] *= G_we;
            G_sum_wtrP[1] *= G_we;
            G_sum_wtrP[2] *= G_we;
            G_sum_wtrP[3] *= G_we;

            G_smoothPath[it][t][0] = G_Ct[0] + G_prePt[0] + G_sum_wtrP[0];
            G_smoothPath[it][t][1] = G_Ct[1] + G_prePt[1] + G_sum_wtrP[1];
            G_smoothPath[it][t][2] = G_Ct[2] + G_prePt[2] + G_sum_wtrP[2];
            G_smoothPath[it][t][3] = G_Ct[3] + G_prePt[3] + G_sum_wtrP[3];
        }
    }

    update_t.clear();
    update_t.resize(originPathV[originPathV.size() - 1].size());
    int t = originPathV.size() - 1;
    vector<cv::Point2f> tempoptim;

    //At this point, if three is the first two, if it is greater than three, it is the first three
    if (iter == 0)
    {
        for (int i = 0; i < originPathV[originPathV.size() - 1].size(); i++)
        {
//            tempoptim.push_back(optimizePathV[t - 1][i] * 0.7 + optimizePathV[t - 2][i] * 0.3);
//            update_t[i] = optimizePathV[t - 1][i] * 0.7 + optimizePathV[t - 2][i] * 0.3 - originPathV[originPathV.size() - 1][i];
            tempoptim.push_back(optimizationPathVV[max_iter-1][t - 1][i]);
            update_t[i] = optimizationPathVV[max_iter-1][t][i] - originPathV[originPathV.size() - 1][i];
        }
    }
    else
    {
        for (int i = 0; i < originPathV[originPathV.size() - 1].size(); i++)
        {
            tempoptim.push_back(optimizationPathVV[max_iter-1][t - 1][i]);
            update_t[i] = optimizationPathVV[max_iter-1][t][i] - originPathV[originPathV.size() - 1][i];
        }
    }

    optimizePathV = optimizationPathVV[max_iter-1];
    optimizationPathVV.push_back(optimizationPathVV[max_iter-1]);





    t = G_optimizePath.size() - 1;
    vector<double> G_tempoptim(4,0.0);
    if (iter == 0)
    {
        G_tempoptim[0] = G_smoothPath[max_iter-1][t][0] ;
        G_tempoptim[1] = G_smoothPath[max_iter-1][t][1] ;
        G_tempoptim[2] = G_smoothPath[max_iter-1][t][2] ;
        G_tempoptim[3] = G_smoothPath[max_iter-1][t][3] ;
        G_update_t[0] = G_smoothPath[max_iter-1][t][0]  - G_originPath[G_originPath.size() - 1][0];
        G_update_t[1] = G_smoothPath[max_iter-1][t][1]  - G_originPath[G_originPath.size() - 1][1];
        G_update_t[2] = G_smoothPath[max_iter-1][t][2]  - G_originPath[G_originPath.size() - 1][2];
        G_update_t[3] = G_smoothPath[max_iter-1][t][3]  - G_originPath[G_originPath.size() - 1][3];
Video_smooth<< G_smoothPath[max_iter-1][t][0] << " " << G_smoothPath[max_iter-1][t][1] << " " << G_smoothPath[max_iter-1][t][2] <<" " << G_smoothPath[max_iter-1][t][3]<< endl;
    }
    else
    {
        G_tempoptim[0] = G_smoothPath[max_iter-1][t][0] ;
        G_tempoptim[1] = G_smoothPath[max_iter-1][t][1] ;
        G_tempoptim[2] = G_smoothPath[max_iter-1][t][2] ;
        G_tempoptim[3] = G_smoothPath[max_iter-1][t][3] ;
        G_update_t[0] = G_smoothPath[max_iter-1][t][0]  - G_originPath[G_originPath.size() - 1][0];
        G_update_t[1] = G_smoothPath[max_iter-1][t][1]  - G_originPath[G_originPath.size() - 1][1];
        G_update_t[2] = G_smoothPath[max_iter-1][t][2]  - G_originPath[G_originPath.size() - 1][2];
        G_update_t[3] = G_smoothPath[max_iter-1][t][3]  - G_originPath[G_originPath.size() - 1][3];
Video_smooth<< G_smoothPath[max_iter-1][t][0] << " " << G_smoothPath[max_iter-1][t][1] << " " << G_smoothPath[max_iter-1][t][2] <<" " << G_smoothPath[max_iter-1][t][3]<< endl;
    }
    G_optimizePath[G_optimizePath.size()-1] = G_tempoptim;
    G_smoothPath.push_back(G_smoothPath[max_iter-1]);

    iter++;
}



//void FastFeature(cv::Mat src, vector<cv::Point2f> &pp1){
//    vector<KeyPoint> tmpKeypoint;
//    vector<KeyPoint> tmpkpp;

//    int img_width = src.cols;
//    int img_height = src.rows;
//    double cutw = img_width / 3;
//    double cuth = img_height / 2;
//    int threshold = 100;

//    Mat img1 = src(Rect(0, 0, cutw, cuth));
//    Mat img2 = src(Rect(cutw, 0, cutw, cuth));
//    Mat img3 = src(Rect(cutw * 2, 0, img_width - cutw * 2, cuth));
//    Mat img4 = src(Rect(0, cuth, cutw,img_height-cuth));
//    Mat img5 = src(Rect(cutw, cuth, cutw, img_height - cuth));
//    Mat img6 = src(Rect(cutw * 2, cuth, img_width - cutw * 2, img_height - cuth));

//    while (true)
//    {
//        FastFeatureDetector fast(threshold);
//        fast.detect(img1, tmpkpp);
//        if (tmpkpp.size() < 100)
//            threshold -= 10;
//        else if (threshold<5)
//            break;
//        else if(tmpkpp.size() >=100)
//            break;
//    }
//    for (int i = 0; i < tmpkpp.size(); i++)
//        tmpKeypoint.push_back(tmpkpp[i]);
//    tmpkpp.clear();

//    while (true)
//    {
//        FastFeatureDetector fast(threshold);
//        fast.detect(img2, tmpkpp);
//        if (tmpkpp.size() < 100)
//            threshold -= 10;
//        else if (threshold<5)
//            break;
//        else if(tmpkpp.size() >= 100)
//            break;
//    }
//    for (int i = 0; i < tmpkpp.size(); i++){
//        tmpkpp[i].pt.x += (int)cutw;
//        tmpKeypoint.push_back(tmpkpp[i]);
//    }

//    tmpkpp.clear();

//    while (true)
//    {
//        FastFeatureDetector fast(threshold);
//        fast.detect(img3, tmpkpp);
//        if (tmpkpp.size() < 100)
//            threshold -= 10;
//        else if (threshold<5)
//            break;
//        else if (tmpkpp.size() >= 100)
//            break;
//    }
//    for (int i = 0; i < tmpkpp.size(); i++){
//        tmpkpp[i].pt.x += (int)cutw*2;
//        tmpKeypoint.push_back(tmpkpp[i]);
//    }
//    tmpkpp.clear();

//    while (true)
//    {
//        FastFeatureDetector fast(threshold);
//        fast.detect(img4, tmpkpp);
//        if (tmpkpp.size() < 100)
//            threshold -= 10;
//        else if (threshold<5)
//            break;
//        else if (tmpkpp.size() >= 100)
//            break;
//    }
//    for (int i = 0; i < tmpkpp.size(); i++){
//        tmpkpp[i].pt.y += (int)cuth;
//        tmpKeypoint.push_back(tmpkpp[i]);
//    }
//    tmpkpp.clear();

//    while (true)
//    {
//        FastFeatureDetector fast(threshold);
//        fast.detect(img5, tmpkpp);
//        if (tmpkpp.size() < 100)
//            threshold -= 10;
//        else if (threshold<5)
//            break;
//        else if (tmpkpp.size() >= 100)
//            break;
//    }
//    for (int i = 0; i < tmpkpp.size(); i++){
//        tmpkpp[i].pt.y += (int)cuth;
//        tmpkpp[i].pt.x += (int)cutw;
//        tmpKeypoint.push_back(tmpkpp[i]);
//    }
//    tmpkpp.clear();

//    while (true)
//    {
//        FastFeatureDetector fast(threshold);
//        fast.detect(img6, tmpkpp);
//        if (tmpkpp.size() < 100)
//            threshold -= 10;
//        else if (threshold<5)
//            break;
//        else if (tmpkpp.size() >= 100)
//            break;
//    }
//    for (int i = 0; i < tmpkpp.size(); i++){
//        tmpkpp[i].pt.y += (int)cuth;
//        tmpkpp[i].pt.x += (int)cutw*2;
//        tmpKeypoint.push_back(tmpkpp[i]);
//    }
//    tmpkpp.clear();

//    for (int i = 0; i <tmpKeypoint.size(); i++)
//        pp1.push_back(tmpKeypoint[i].pt);
//}



void QuickSort(vector<float> &arr, int left, int right){
      int i = left, j = right;
      float tmp;
      float pivot = arr[(left + right) / 2];

      while(i<=j){
          while(arr[i]<pivot)
              i++;
          while(arr[j]>pivot)
              j--;
          if(i<=j){
              tmp = arr[i];
              arr[i] = arr[j];
              arr[j] = tmp;
              i++;
              j--;
          }
      }
      if(left<j)QuickSort(arr,left,j);
      if(i<right)QuickSort(arr,i,right);
}

void MeshFlowVS::meshWarp(const cv::Mat src, cv::Mat &dst, const Mesh &m1, const Mesh &m2){
    dst = cv::Mat::zeros( src.size(), CV_8UC3 );
    for(int i=1; i<m1.meshHeight; i++){
        for(int j=1; j<m1.meshWidth; j++){
            cv::Point2f p0 = m1.getVertex(i-1,j-1);
            cv::Point2f p1 = m1.getVertex(i-1,j);
            cv::Point2f p2 = m1.getVertex(i,j-1);
            cv::Point2f p3 = m1.getVertex(i,j);

            cv::Point2f q0 = m2.getVertex(i-1, j-1);
            cv::Point2f q1 = m2.getVertex(i-1, j);
            cv::Point2f q2 = m2.getVertex(i, j-1);
            cv::Point2f q3 = m2.getVertex(i, j);

            Quad quad1(p0,p1,p2,p3);
            Quad quad2(q0,q1,q2,q3);
            quadWarp(src,dst,quad1,quad2);
        }
    }
}


void MeshFlowVS::quadWarp( const cv::Mat src, cv::Mat &dst, const Quad &q1, const Quad &q2){
    int minx = max(0, (int)q2.getMinX());
    int maxx = min(dst.cols-1, (int)q2.getMaxX());
    int miny = max(0, (int)q2.getMinY());
    int maxy = min(dst.rows-1, (int)q2.getMaxY());

    vector<cv::Point2f> source(4);
    vector<cv::Point2f> target(4);
    source[0] = q1.V00;
    source[1] = q1.V01;
    source[2] = q1.V10;
    source[3] = q1.V11;

    target[0] = q2.V00;
    target[1] = q2.V01;
    target[2] = q2.V10;
    target[3] = q2.V11;
    cv::Mat H = cv::findHomography(source, target, 0);
    for (int i = miny; i < maxy; i ++) {
        for (int j = minx; j < maxx; j ++) {
            double X = H.at<double>(0, 0) * j + H.at<double>(0, 1) * i + H.at<double>(0, 2);
            double Y = H.at<double>(1, 0) * j + H.at<double>(1, 1) * i + H.at<double>(1, 2);
            double W = H.at<double>(2, 0) * j + H.at<double>(2, 1) * i + H.at<double>(2, 2);
            W = W ? 1.0 / W : 0;
            X = (int)X*W;
            Y = (int)Y*W;
            cv::Vec3b color = src.at<cv::Vec3b>(cv::Point(j,i));
            if(X>=0 && X<src.cols && Y>=0 && Y< src.rows)
            {
                dst.at<cv::Vec3b>(cv::Point(X,Y)) = color;
            }
        }
    }

}




