//
//  eye.cpp
//  HelloOpenCV
//
//  Created by Chih-Sheng Wang on 9/29/13.
//  Copyright (c) 2013 huangtw. All rights reserved.
//
#include <iostream>
#include "eye.h"
 eye::eye(cv::Mat mb, cv::vector< cv::Point_<int> > V){
    this->mb = mb;
    this->V = V;
}
cv::Mat eye::getMB(){
    //if (this->mb!=NULL) {
        return mb;
    //}
    //else{
    //    perror("eye.mb == NULL\n");
   // }

}
cv::Mat eye::changeEyeColor(cv::Mat& img, int hueThreshold, int targetHue){
    cv::Mat hsvImage;
    cv::cvtColor(img, hsvImage, CV_BGR2HSV);

    cv::Mat mask;
    //select change zone
    cv::inRange(hsvImage, cv::Scalar(0, 0, 0), cv::Scalar(200, 200, 200), mask);

    hsvImage.setTo(cv::Scalar(60,255,255),mask);

    cv::Mat result;
    cv::cvtColor(hsvImage, result, CV_HSV2BGR);

    return result;
}
