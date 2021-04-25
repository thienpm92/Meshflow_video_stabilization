#include "meshflow.h"
#include "meshgrid.h"
#include "mesh.h"
#include "quad.h"
#include <string>
#include <stdio.h>
#include <omp.h>
#include <time.h>

using namespace std;

MeshFlowVS::MeshFlowVS(string src_path,string dst_path)
{
    videopath = src_path;
    savevideo = dst_path;
    optimizationPathVV.resize(max_iter);
}

MeshFlowVS::~MeshFlowVS()
{

}

void MeshFlowVS::startRun()
{
    ofstream OrgPath("OrgPath.txt");
    ofstream SmoothPath("SmoothPath.txt");
    char buffer[1024] = {};
    int tt = 1;
    int iter = 0;
    clock_t start, finish;

    frameNum = 0;

    vector<cv::Point2f> optpath;  //all node zero value
    vector<cv::Point2f> originPath;   //all node zero value
    cv::Mat	tempFrame;
    cv::Mat gray;
    cv::Mat prevgray;
    cv::Mat prev,cur;
    cv::VideoCapture cap(videopath);
    cap>>prev;
    cv::VideoWriter write_video(savevideo, CV_FOURCC('D', 'I', 'V', 'X'), 30, cvSize(prev.cols, prev.rows), true);
    int max_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
    cout<<"max frame "<<max_frames<<endl;

    cv::cvtColor(prev, prevgray, CV_BGR2GRAY);
    frameV.push_back(prev);

    img_width = prevgray.cols;
    img_height = prevgray.rows;
    subImageHeight = img_height / 10;
    subImageWidth = img_width / 10;
    double weight =1;
    double gridWidth = img_width/16.0;
    double gridHeight = img_height/16.0;

    initMeshFlow(img_width,img_height);
    initOriginPath();   //originPathV create all node 0 value

    ofstream Video_traject("Video_traject.txt");
    ofstream Video_smooth("Video_smooth.txt");
    ofstream Video_parameter("Video_parameter.txt");
    cv::Mat smooth_T(2, 3, CV_64F);
    G_smoothPath.resize(max_iter);
    G_update_t.resize(4,0.0);
    store.resize(4,0.0);
    double traject_a = 0;
    double traject_s = 1;
    double traject_x = 0;
    double traject_y = 0;

    while(true)
    {     
        cap >> cur;
        if(cur.data==NULL)
            break;
        cout<<frameNum<<endl;
        frameNum++;
        vector<cv::Point2f> pp1;
        vector<cv::Point2f> pp2;
        vector<uchar> status;
        vector<float> err;
        m_warpedemesh = new Mesh(img_height, img_width, 1.0*m_quadWidth, 1.0*m_quadHeight);
        if(frameV.size()==40)
        {
            vector<cv::Mat>::iterator it = frameV.begin();
            frameV.erase(it);
            frameV.push_back(cur);
            vector<vector<cv::Point2f>>::iterator it1 = originPathV.begin();	//Not only to delete the frame, but also to delete the homography matrix and, Ct
            originPathV.erase(it1);
            vector<vector<cv::Point2f>>::iterator it2 = optimizePathV.begin();  //At this point, the number of original paths is smaller than the number of frames,
                                                                           // and the optimized path obtained from the last iteration is reduced by one.
            optimizePathV.erase(it2);

            vector<vector<double>>::iterator it3 = G_originPath.begin();
            G_originPath.erase(it3);
            vector<vector<double>>::iterator it4 = G_optimizePath.begin();
            G_optimizePath.erase(it4);
            vector<vector<double>>::iterator it5 = G_orgParameter.begin();
            G_orgParameter.erase(it5);
        }
        else{
            frameV.push_back(cur);
        }

        cv::cvtColor(cur, gray, CV_BGR2GRAY);
        cv::goodFeaturesToTrack(prevgray, pp1, max_corners, quality_level, minDistance);
        cv::calcOpticalFlowPyrLK(prevgray, gray, pp1, pp2,status , err);

        vector<cv::Point2f> prev_temp, cur_temp;
        for (int i = 0; i < status.size(); i++) {
            if (status[i]) {
                prev_temp.push_back(pp1[i]);
                cur_temp.push_back(pp2[i]);
            }
        }


        /*tam luot qua Fast Feature
          TRACKING FEATURES  */
        glFeatures.clear();
        glFeatures2.clear();
        int ptCount = status.size();//
        cv::Mat p1(ptCount, 2, CV_32F);
        cv::Mat p2(ptCount, 2, CV_32F);
        for (int j = 0; j < ptCount; j++)
        {
            p1.at<float>(j, 0) = pp1[j].x;
            p1.at<float>(j, 1) = pp1[j].y;
            p2.at<float>(j, 0) = pp2[j].x;
            p2.at<float>(j, 1) = pp2[j].y;
        }
        cv::Mat m_Fundamental;
        vector<uchar> m_RANSACStatus;
        m_Fundamental = cv::findFundamentalMat(p1, p2, m_RANSACStatus, cv::FM_RANSAC);
        for (int j = 0; j < ptCount; j++)
        {
            if (m_RANSACStatus[j] == 0)
            {
                status[j] = 0;
            }
        }
        for (int j = 0; j < pp2.size(); j++)
        {
            if (status[j] == 0 || (pp2[j].x <= 0 || pp2[j].y <= 0 || pp2[j].x >= img_width - 1 || pp2[j].y >= img_height - 1))
                continue;
            glFeatures2.push_back(pp2[j]);
            glFeatures.push_back(pp1[j]);
        }
        /*DISTRIBUTE MOTION VECTORS*/
        vector<cv::Point2f> motion;
        motion.resize(glFeatures2.size());
        m_globalHomography = cv::findHomography(cv::Mat(glFeatures), cv::Mat(glFeatures2));
        homographV.push_back(m_globalHomography);

        cv::Mat T = estimateRigidTransform(prev_temp, cur_temp, false);
        double org_dx = T.at<double>(0, 2);
        double org_dy = T.at<double>(1, 2);
        double org_da = atan2(T.at<double>(1, 0), T.at<double>(0, 0));
        double org_ds = T.at<double>(0, 0)/cos(org_da);
Video_parameter << org_dx << " " << org_dy << " " << org_da << " " << org_ds << endl;
        store[0]=org_dx; store[1]=org_dy; store[2]=org_da; store[3]=org_ds;
        G_orgParameter.push_back(store);
        traject_x += G_orgParameter[G_orgParameter.size()-1][0];
        traject_y += G_orgParameter[G_orgParameter.size()-1][1];
        traject_a += G_orgParameter[G_orgParameter.size()-1][2];
        traject_s *= G_orgParameter[G_orgParameter.size()-1][3];
Video_traject << traject_x << " " << traject_y << " " << traject_a << " " << traject_s << endl;
        store[0]=traject_x; store[1]=traject_y; store[2]=traject_a; store[3]=traject_s;
        G_originPath.push_back(store);


        for (int i = 0; i < glFeatures2.size(); i++)
            motion[i] = glFeatures2[i] - glFeatures[i]+ glFeatures2[i] - Trans(m_globalHomography, glFeatures[i]);

        double a =  m_globalHomography.at<double>(0, 2);
        double b = m_globalHomography.at<double>(1, 2);
        double c = m_globalHomography.at<double>(2, 2);
        cv::Point2f globalMotion(a/c, b/c);
        DistributeMotion2MeshVertexes_MedianFilter(glFeatures2, motion, globalMotion);
        SpatialMedianFilter2();
        computeOriginPath();

        if (tt == 1)
        {
            vector<cv::Point2f> optpath = originPathV[originPathV.size() - 1];
            vector<cv::Point2f> originpath = originPathV[originPathV.size() - 2];
            m_previousFrame = frameV[frameV.size() - 2];
            m_currentFrame = frameV[frameV.size() - 1];
            wrp_currentFrame = cv::Mat::zeros( m_currentFrame.size(), CV_8UC3 );
            ViewSynthesis(optpath, originpath);
            vector<cv::Point2f> srcNodes, dstNodes;
            for (int i = 0; i <m_meshHeight; i++)  {
                for (int j = 0; j < m_meshwidth; j++)  {
                     cv::Point2f pt1 = m_mesh->getVertex(i,j);
                     cv::Point2f pt2 = m_warpedemesh->getVertex(i,j);
                     srcNodes.push_back(pt1);
                     dstNodes.push_back(pt2);
                }
            }

            MeshGrid WarpingMethod(img_width, img_height, gridWidth, gridHeight, weight);
            WarpingMethod.SetControlPts(srcNodes,dstNodes);
            WarpingMethod.Solve();
            wrp_currentFrame = WarpingMethod.Warping(prev);

//            meshWarp(m_previousFrame, wrp_currentFrame, *m_mesh, *m_warpedemesh);

            cv::Mat drawImg = m_currentFrame.clone();
            m_warpedemesh->drawMesh( drawImg);
            sprintf(buffer, "output%05d.jpg", frameNum);
            cv::imwrite(buffer, prev);
            write_video.write(wrp_currentFrame);
            tt++;

            double smooth_dx = (G_originPath[G_orgParameter.size()-1][0] );
            double smooth_dy = (G_originPath[G_orgParameter.size()-1][1] );
            double smooth_da = (G_originPath[G_orgParameter.size()-1][2] );
            double smooth_ds = (G_originPath[G_orgParameter.size()-1][3] );
Video_smooth<< G_originPath[G_originPath.size()-1][0] << " " << G_originPath[G_originPath.size()-1][1] << " " << G_originPath[G_originPath.size()-1][2] <<" " << G_originPath[G_originPath.size()-1][3]<< endl;
            smooth_T.at<double>(0, 0) = cos(smooth_da);
            smooth_T.at<double>(0, 1) = -sin(smooth_da);
            smooth_T.at<double>(1, 0) = sin(smooth_da);
            smooth_T.at<double>(1, 1) = cos(smooth_da);
            smooth_T.at<double>(0, 2) = smooth_dx;
            smooth_T.at<double>(1, 2) = smooth_dy;

//            int step = 8;
//            for(int i = 0;i<=17;i++){
//                OrgPath<<originpath[step*i].x<<" ";
//                SmoothPath<<optpath[step*i].x<<" ";
//            }
//            OrgPath<<endl;
//            SmoothPath<<endl;
        }
        else
        {
            Jacobicompute(Video_smooth);
            vector<cv::Point2f> optpath = optimizePathV[optimizePathV.size() - 1];
            vector<cv::Point2f> originpath = originPathV[originPathV.size() - 1];
            ViewSynthesis(optpath, originpath);

            vector<cv::Point2f> srcNodes, dstNodes;
            for (int i = 0; i <m_meshHeight; i++)  {
                for (int j = 0; j < m_meshwidth; j++)  {
                     cv::Point2f pt1 = m_mesh->getVertex(i,j);
                     cv::Point2f pt2 = m_warpedemesh->getVertex(i,j);
                     srcNodes.push_back(pt1);
                     dstNodes.push_back(pt2);
                }
            }

//            MeshGrid WarpingMethod(img_width, img_height, gridWidth, gridHeight, weight);
//            WarpingMethod.SetControlPts(srcNodes,dstNodes);
//            WarpingMethod.Solve();
//            wrp_currentFrame = WarpingMethod.WarpingGlobal(prev);
//            meshWarp(prev, wrp_currentFrame, *m_mesh, *m_warpedemesh);

            cv::Mat H = cv::findHomography(srcNodes, dstNodes, 0);
            wrp_currentFrame = cv::Mat::zeros( prev.size(), CV_8UC3 );
            for (int i = 0; i < prev.rows; i ++) {
                for (int j = 0; j < prev.cols; j ++) {
                    double X = H.at<double>(0, 0) * j + H.at<double>(0, 1) * i + H.at<double>(0, 2);
                    double Y = H.at<double>(1, 0) * j + H.at<double>(1, 1) * i + H.at<double>(1, 2);
                    double W = H.at<double>(2, 0) * j + H.at<double>(2, 1) * i + H.at<double>(2, 2);
                    W = W ? 1.0 / W : 0;
                    X = (int)X*W;
                    Y = (int)Y*W;
                    cv::Vec3b color = prev.at<cv::Vec3b>(cv::Point(j,i));
                    if(X>=0 && X<prev.cols && Y>=0 && Y< prev.rows)
                    {
                        wrp_currentFrame.at<cv::Vec3b>(cv::Point(X,Y)) = color;
                    }
                }
            }

//            cv::Mat drawImg = m_currentFrame.clone();
//            m_warpedemesh->drawMesh( drawImg);
            sprintf(buffer, "output%05d.jpg", frameNum);
            cv::imwrite(buffer, prev);
            write_video.write(wrp_currentFrame);

            double smooth_dx = (G_orgParameter[G_orgParameter.size()-1][0] + G_update_t[0]);
            double smooth_dy = (G_orgParameter[G_orgParameter.size()-1][1] + G_update_t[1]);
            double smooth_da = (G_orgParameter[G_orgParameter.size()-1][2] + G_update_t[2]);
            double smooth_ds = (G_orgParameter[G_orgParameter.size()-1][3] * G_update_t[3]);
            smooth_T.at<double>(0, 0) = cos(smooth_da);
            smooth_T.at<double>(0, 1) = -sin(smooth_da);
            smooth_T.at<double>(1, 0) = sin(smooth_da);
            smooth_T.at<double>(1, 1) = cos(smooth_da);
            smooth_T.at<double>(0, 2) = smooth_dx;
            smooth_T.at<double>(1, 2) = smooth_dy;

            int step = 3;
            for(int i = 0;i<=16;i++){
                OrgPath<<originpath[step*i].y<<" ";
                SmoothPath<<optpath[step*i].y<<" ";
            }
            OrgPath<<endl;
            SmoothPath<<endl;
        }

//        warpAffine(prev, wrp_currentFrame, smooth_T, prev.size());
//        write_video.write(wrp_currentFrame);

        gray.copyTo(prevgray);
        cur.copyTo(prev);
        //        start = clock();
        //        finish = clock();
        //        double duration = (double)(finish - start) / CLOCKS_PER_SEC;
        //        cout << " time " << duration << endl;
//        m_currentFrame = frameV[frameV.size() - 1];
    }
    cout<<"done"<<endl;




        //////////////////////////

//        localFeatures.clear();
//        localFeatures.resize(100);
//        for (int k = 0; k < glFeatures2.size(); k++)
//        {
//            int t = getSubIndex(glFeatures2[k]);
//            localFeatures[t].push_back(glFeatures2[k]);
//        }
//        int flag = 0;
//        subImages.clear();
//        Frame2subFrames(frameV[frameV.size() - 1], subImages);

//        for (int k = 0; k < localFeatures.size(); k++)
//        {
//            if (localFeatures[k].size() < localmincount)
//            {
//                int threshold = 100;
//                while (true)
//                {
//                    tmpkp.clear();
//                    FastFeatureDetector fast(threshold);
//                    cvtColor(subImages[k], subgray, CV_BGR2GRAY);
//                    fast.detect(subgray, tmpkp);
//                    if (threshold < 10)
//                        break;

//                    if (tmpkp.size() < localmincount)
//                        threshold -= 5;
//                    else
//                        break;
//                }


//                vector<Point2f> tmp;
//                for (int i = 0; i < tmpkp.size(); i++)
//                    tmp.push_back(tmpkp[i].pt);


//                //#pragma omp  parallel for
//                for (int j = 0; j < tmp.size(); j++)
//                {
//                    bool flag = true;
//                    for (int m = 0; m < localFeatures[k].size(); m++)
//                    {
//                        //选择与原来由光流法计算出的特征点相近的特征点
//                        if ((tmp[j].x - localFeatures[k][m].x)*(tmp[j].x - localFeatures[k][m].x) + (tmp[j].y - localFeatures[k][m].y)*(tmp[j].y - localFeatures[k][m].y) < 10)
//                        {
//                            flag = false;
//                            break;
//                        }
//                    }

//                    if (flag == true)
//                    {
//                        Point2f p(tmp[j].x + (k % 10)*subImageWidth, tmp[j].y + (k / 10)*subImageHeight);
//                        glFeatures2.push_back(p);
//                    }
//                }
//            }
//        }
//        subImages.clear();

//        gray.copyTo(graypre);
//        std::swap(glFeatures, glFeatures2);
//    }

}
