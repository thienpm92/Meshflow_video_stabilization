#include "meshflow.h"
#include "mesh.h"
#include "quad.h"
int main(int argc, char *argv[])
{
    string videopath = "/media/eric/DATA/SSU research/Data set/video_test_case/regular_girl_2.avi";
    string savevideo = "meshflow_outdoor.avi";
    ofstream OrgPath ("OrgPath.txt");

    MeshFlowVS videostab(videopath,savevideo);
    cout<<"START"<<endl;

    videostab.startRun();



    cout<<"FINISH"<<endl;
    return 0;
}
