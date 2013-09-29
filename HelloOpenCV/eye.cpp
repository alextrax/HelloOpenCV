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
    cv::Mat hsvImage = img.clone();
    cv::cvtColor(img, hsvImage, CV_BGR2HSV);
    
    std::vector<cv::Mat> channels;
    
    cv::split(hsvImage, channels);
    cv::Mat hue = channels[0];
    cv::Mat dest;
    cv::Mat temp = img.clone();
    
    //select change zone
    cv::inRange(hsvImage, cv::Scalar(0, 0, 0), cv::Scalar(200, 200, 200), dest);
    
    cv::merge(channels, temp);
    temp.setTo(cv::Scalar(60,255,255),dest);
    cv::split(temp, channels);
    cv::merge(channels, dest);
    cv::cvtColor(dest, img, CV_HSV2BGR);
    return img;
    
}
