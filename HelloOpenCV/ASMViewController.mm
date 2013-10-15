//
//  ASMViewController.m
//  HelloOpenCV
//
//  Created by d71941 on 9/14/13.
//  Copyright (c) 2013 huangtw. All rights reserved.
//

#import "ASMViewController.h"
#import "asmmodel.h"
#import "eye.h"

@interface ASMViewController ()

@end

@implementation ASMViewController
@synthesize image = _image;

void showNumberOnImg(Mat& src, Mat& dst, const vector< cv::Point >& vP)
{
    if (&src != &dst)
    {
        dst = src.clone();
    }

    for (uint i=0;i<vP.size();i++){
        //27~31(outter), 68~71(inner): right eye
        //32~36(outter), 72~75(inner): left eye
        //48~66: mouth
        char text[8];
        sprintf(text, "%d", i);
        putText(dst, text, vP[i], CV_FONT_HERSHEY_SIMPLEX, 0.25, cv::Scalar(25, 50, 255), 1, CV_AA);
    }
}

void showHueHistogram(Mat& img)
{
    Mat hsvImg;
    cv::cvtColor(img, hsvImg, CV_BGR2HSV);

    // Quantize the hue to 180 levels
    int hbins = 180;
    int histSize[] = {hbins};
    // hue varies from 0 to 179
    float hranges[] = { 0, 180 };
    const float* ranges[] = {hranges};
    cv::MatND hist;
    int channels[] = {0};

    calcHist(&hsvImg,
             1,
             channels,
             Mat(), // do not use mask
             hist,
             1,
             histSize,
             ranges,
             true, // the histogram is uniform
             false);

    for (int i = 0; i < 180; i++)
    {
        float val = hist.at<float>(i);
        NSLog(@"%d: %f", i, val);
    }
}

void mixChannel(Mat& src, Mat& dst, float r, float g, float b)
{
    Mat mix;
    mix = src.clone();
    
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            int value = b*src.at<cv::Vec3b>(i,j)[0] + g*src.at<cv::Vec3b>(i,j)[1] + r*src.at<cv::Vec3b>(i,j)[2];
            if(value > 255)
            {
                value = 255;
            }
            else if(value < 0)
            {
                value = 0;
            }
            
            mix.at<cv::Vec3b>(i,j)[0] = value;
            mix.at<cv::Vec3b>(i,j)[1] = value;
            mix.at<cv::Vec3b>(i,j)[2] = value;
        }
    }

    cv::cvtColor(mix, dst, CV_BGR2GRAY);
}

void maskByContour(Mat& dst, cv::Rect& roiRect, vector<cv::Point> contour)
{
    vector<cv::Point> contourRelative = contour;
    for (int i = 0; i < contourRelative.size(); i++)
    {
        contourRelative[i].x = contourRelative[i].x - roiRect.x;
        contourRelative[i].y = contourRelative[i].y - roiRect.y;
    }

    vector< vector<cv::Point> > contours;
    contours.push_back(contourRelative);

    dst = Mat::zeros(roiRect.height, roiRect.width, CV_8UC1);
    cv::drawContours(dst, contours, -1, cv::Scalar(255), CV_FILLED);
}

void maskByBackgroundHue(Mat& src, Mat& dst)
{
    Mat hsvImg;
    cv::cvtColor(src, hsvImg, CV_BGR2HSV);
    
    //NSLog(@"%d", hsvImg.at<cv::Vec3b>(0,0)[0]);
    int skinHue = hsvImg.at<cv::Vec3b>(0,0)[0];

    cv::inRange(hsvImg, cv::Scalar(skinHue - 5, 0, 0), cv::Scalar(skinHue + 5, 255, 255), dst);
    dst = Mat::ones(dst.size(), dst.type()) * 255 - dst; //invert
}

void maskByBinaryImage(Mat& src, Mat& dst)
{
    if (&src != &dst)
    {
        dst = src.clone();
    }
    
    mixChannel(dst, dst, -0.7, 2.0, -0.3);
    cv::threshold(dst, dst, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    dst = Mat::ones(dst.size(), dst.type()) * 255 - dst; //invert
}

void applyCannyFilter(Mat& src, Mat& dst)
{
    Mat temp;
    double threshold;
    
    cv::cvtColor(src, dst, CV_BGR2GRAY);
    threshold = cv::threshold(dst, temp, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    cv::Canny(dst, dst, threshold*0.5, threshold);
}

void changeEyeColor(Mat& src, Mat& dst, const vector< cv::Point >& vP)
{
    if (&src != &dst)
    {
        dst = src.clone();
    }

    vector< cv::Point > eyeContour;
    
    for (int i = 27; i <= 31; i++)
    {
        eyeContour.push_back(vP[i]);
    }
    
    cv::Rect mouthRect = boundingRect(eyeContour);
    
    //mouthRect.x = mouthRect.x - mouthRect.width*0.1;
    //mouthRect.width = mouthRect.width*1.2;
    //mouthRect.y = mouthRect.y - mouthRect.height*0.25;
    //mouthRect.height = mouthRect.height*1.5;
    Mat roi = Mat(dst, mouthRect); //Region of intrest

    Mat mask;
    //maskByBackgroundHue(roi, mask);
    //maskByContour(mask, mouthRect, mouthContour);
    maskByBinaryImage(roi, mask);

    
    //Use the following two lines to show mask
    //cv::cvtColor(mask, roi, CV_GRAY2BGR);
    //return;

    cv::Mat hsvImg;
    cv::cvtColor(roi, hsvImg, CV_BGR2HSV);
    
    vector<cv::Mat> matsForEachChannel;
    cv::split(hsvImg, matsForEachChannel);
    matsForEachChannel[0].setTo(cv::Scalar(30), mask);
    cv::merge(matsForEachChannel, hsvImg);
    
    cv::cvtColor(hsvImg, roi, CV_HSV2BGR);
}

void changeLipsColor(Mat& src, Mat& dst, const vector< cv::Point >& vP)
{
    if (&src != &dst)
    {
        dst = src.clone();
    }

    vector< cv::Point > mouthContour;

    for (int i = 48; i <= 59; i++)
    {
        mouthContour.push_back(vP[i]);
    }

    cv::Rect mouthRect = boundingRect(mouthContour);
    
    //mouthRect.x = mouthRect.x - mouthRect.width*0.1;
    //mouthRect.width = mouthRect.width*1.2;
    //mouthRect.y = mouthRect.y - mouthRect.height*0.25;
    //mouthRect.height = mouthRect.height*1.5;
    Mat roi = Mat(dst, mouthRect); //Region of intrest

    Mat mask;
    //maskByBackgroundHue(roi, mask);
    //maskByContour(mask, mouthRect, mouthContour);
    maskByBinaryImage(roi, mask);


    //Use the following two lines to show mask
    //cv::cvtColor(mask, roi, CV_GRAY2BGR);
    //return;
    
    cv::Mat hsvImg;
    cv::cvtColor(roi, hsvImg, CV_BGR2HSV);

    vector<cv::Mat> matsForEachChannel;
    cv::split(hsvImg, matsForEachChannel);
    matsForEachChannel[0].setTo(cv::Scalar(30), mask);
    cv::merge(matsForEachChannel, hsvImg);

    cv::cvtColor(hsvImg, roi, CV_HSV2BGR);
}

+ (UIImage *)rotateImage:(UIImage *)image toSize:(CGSize)targetSize
{
    UIImage *sourceImage = image;
    CGFloat width, height, targetWidth, targetHeight;

    CGImageRef imageRef = [sourceImage CGImage];

    if (sourceImage.imageOrientation == UIImageOrientationUp || sourceImage.imageOrientation == UIImageOrientationDown) {
        width = CGImageGetWidth(imageRef);
        height = CGImageGetHeight(imageRef);
    }
    else
    {
        width = CGImageGetHeight(imageRef);
        height = CGImageGetWidth(imageRef);
    }

    CGFloat ratio = width/height;
    if (width > height)
    {
        targetWidth = MAX(targetSize.width, targetSize.height);
        targetHeight = targetWidth/ratio;
    }
    else
    {
        targetHeight = MAX(targetSize.width, targetSize.height);
        targetWidth = targetHeight*ratio;
    }
    
    CGBitmapInfo bitmapInfo = CGImageGetBitmapInfo(imageRef);
    CGColorSpaceRef colorSpaceInfo = CGImageGetColorSpace(imageRef);

    if (bitmapInfo == kCGImageAlphaNone)
    {
        bitmapInfo = kCGImageAlphaNoneSkipLast;
    }

    CGContextRef bitmap;

    bitmap = CGBitmapContextCreate(NULL, targetWidth, targetHeight, CGImageGetBitsPerComponent(imageRef), CGImageGetBitsPerComponent(imageRef)*targetWidth, colorSpaceInfo, bitmapInfo);

    if (sourceImage.imageOrientation == UIImageOrientationLeft)
    {
        CGContextRotateCTM (bitmap, M_PI_2);
        CGContextTranslateCTM (bitmap, 0, -targetWidth);
    } else if (sourceImage.imageOrientation == UIImageOrientationRight)
    {
        CGContextRotateCTM (bitmap, -M_PI_2);
        CGContextTranslateCTM (bitmap, -targetHeight, 0);
    } else if (sourceImage.imageOrientation == UIImageOrientationUp)
    {
        // NOTHING
    } else if (sourceImage.imageOrientation == UIImageOrientationDown)
    {
        CGContextTranslateCTM (bitmap, targetWidth, targetHeight);
        CGContextRotateCTM (bitmap, -M_PI);
    }

    if (sourceImage.imageOrientation == UIImageOrientationUp || sourceImage.imageOrientation == UIImageOrientationDown)
    {
        CGContextDrawImage(bitmap, CGRectMake(0, 0, targetWidth, targetHeight), imageRef);
    }
    else
    {
        CGContextDrawImage(bitmap, CGRectMake(0, 0, targetHeight, targetWidth), imageRef);
    }
    CGImageRef ref = CGBitmapContextCreateImage(bitmap);
    UIImage* newImage = [UIImage imageWithCGImage:ref];

    CGContextRelease(bitmap);
    CGImageRelease(ref);

    return newImage;
}

- (cv::Mat)cvMatFromUIImage:(UIImage *)image
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to  data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    CGColorSpaceRelease(colorSpace);
    
    cv::Mat m;
    cv::cvtColor(cvMat, m, CV_RGB2BGR);
    
    return m;
}

-(UIImage *)UIImageFromCVMat:(cv::Mat)cvMat
{
    cv::Mat m;

    if (cvMat.channels()==1)
        cv::cvtColor(cvMat, m, CV_GRAY2RGB);
    else
        cv::cvtColor(cvMat, m, CV_BGR2RGB);

    NSData *data = [NSData dataWithBytes:m.data length:m.elemSize()*m.total()];
    CGColorSpaceRef colorSpace;
    
    if (m.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(m.cols,                                 //width
                                        m.rows,                                 //height
                                        8,                                          //bits per component
                                        8 * m.elemSize(),                       //bits per pixel
                                        m.step[0],                            //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    
    
    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return finalImage;
}
- (id)initWithNibName:(NSString *)nibNameOrNil bundle:(NSBundle *)nibBundleOrNil
{
    self = [super initWithNibName:nibNameOrNil bundle:nibBundleOrNil];
    if (self) {
        // Custom initialization
    }
    return self;
}

- (void)viewDidLoad
{
    [super viewDidLoad];

    self.scrollView.delegate = self;
}

- (UIImage*)processImage:(UIImage*)image
{
    // Load image.
    Mat img = [self cvMatFromUIImage:[[self class] rotateImage:image toSize:CGSizeMake(480, 640)]];
    if (img.empty())
    {
        NSLog(@"load image fail");
        return nil;
    }
    
    //Load ASM Model
    StatModel::ASMModel asmModel;
    std::string asmModelPath = [[[NSBundle mainBundle] pathForResource:@"muct76" ofType:@"model"] cStringUsingEncoding:[NSString defaultCStringEncoding]];
    asmModel.loadFromFile(asmModelPath);
    
    //Load face model
    cv::CascadeClassifier faceCascade;
    std::string faceCascadePath = [[[NSBundle mainBundle] pathForResource:@"haarcascade_frontalface_alt" ofType:@"xml"] cStringUsingEncoding:[NSString defaultCStringEncoding]];
    if (!faceCascade.load(faceCascadePath))
    {
        NSLog(@"faceCascade load fail");
        return nil;
    }
    
    // Face detection.
    vector<cv::Rect> faces;
    faceCascade.detectMultiScale(img, faces, 1.2, 2, CV_HAAR_SCALE_IMAGE, cv::Size(60, 60));
    
    // Fit to ASM!
    vector<StatModel::ASMFitResult> fitResult = asmModel.fitAll(img, faces, 0);
    
    Mat mb;
    if (img.channels()==1)
        cv::cvtColor(img, mb, CV_GRAY2RGB);
    else
        mb = img.clone();
    
    for (uint i = 0; i < fitResult.size(); i++){
        vector< Point_<int> > V;
        fitResult[i].toPointList(V);
        //eye eyeTest(mb, V);
        //mb = eyeTest.changeEyeColor(mb, 0, 0);
        //mb = eyeTest.getMB();

        changeLipsColor(mb, mb, V);
        //changeEyeColor(mb, mb, V);
        mb = asmModel.getShapeInfo().drawMarkPointsOnImg(mb, V, true);
        showNumberOnImg(mb, mb, V);
    }

    return [self UIImageFromCVMat:mb];
}

- (void)viewDidAppear:(BOOL)animated
{
    dispatch_queue_t queue = dispatch_queue_create("processing image", NULL);
    dispatch_async(queue, ^{
        UIImage *image = [self processImage:self.image];
        dispatch_async(dispatch_get_main_queue(), ^{
            self.imageView.image = image;
        });
    });
}

- (void)didReceiveMemoryWarning
{
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

- (UIView*)viewForZoomingInScrollView:(UIScrollView *)scrollView
{
    return self.contentView;
}

@end
