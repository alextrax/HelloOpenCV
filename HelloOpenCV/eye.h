//
//  eye.h
//  HelloOpenCV
//
//  Created by Chih-Sheng Wang on 9/29/13.
//  Copyright (c) 2013 huangtw. All rights reserved.
//

#ifndef __HelloOpenCV__eye__
#define __HelloOpenCV__eye__

#include <iostream>
#include <vector>
#include <cstdio>

//#import "ASMViewController.h"
//#import "asmmodel.h"*/
class eye{
     cv::Mat mb;
     cv::vector< cv::Point_<int> > V;
     public:
     eye(cv::Mat mb, cv::vector< cv::Point_<int> >);
     cv::Mat getMB();
     cv::Mat changeEyeColor(cv::Mat&, int hueThreshold, int targetHue);
       
    
};


#endif /* defined(__HelloOpenCV__eye__) */

