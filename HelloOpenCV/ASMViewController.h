//
//  ASMViewController.h
//  HelloOpenCV
//
//  Created by d71941 on 9/14/13.
//  Copyright (c) 2013 huangtw. All rights reserved.
//

#import <UIKit/UIKit.h>

@interface ASMViewController : UIViewController <UINavigationControllerDelegate, UIImagePickerControllerDelegate, UIScrollViewDelegate>

@property (weak, nonatomic) IBOutlet UIScrollView *scrollView;
@property (weak, nonatomic) IBOutlet UIImageView *imageView;
@property (strong, nonatomic) UIImage *image;
@end
