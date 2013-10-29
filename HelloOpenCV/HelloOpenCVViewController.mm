//
//  HelloOpenCVViewController.m
//  HelloOpenCV
//
//  Created by d71941 on 9/12/13.
//  Copyright (c) 2013 huangtw. All rights reserved.
//

#import "HelloOpenCVViewController.h"
#import "ASMViewController.h"
#import "asmmodel.h"

@interface HelloOpenCVViewController ()

@end

@implementation HelloOpenCVViewController

- (void)viewDidLoad
{
    [super viewDidLoad];
}

- (void)viewDidAppear:(BOOL)animated
{
    [super viewDidAppear:animated];
}

- (void)imagePickerController:(UIImagePickerController *)picker didFinishPickingMediaWithInfo:(NSDictionary *)info
{
    UIStoryboard *mainStoryboard = [UIStoryboard storyboardWithName:@"MainStoryboard" bundle: nil];
    ASMViewController *asmViewController = [mainStoryboard instantiateViewControllerWithIdentifier:@"asm"];


    asmViewController.image = [info objectForKey:UIImagePickerControllerOriginalImage];
    asmViewController.useSTASM = YES;

    [picker dismissViewControllerAnimated:YES completion:nil];
    [self.navigationController pushViewController:asmViewController animated:YES];
}
- (IBAction)onPhotosClicked:(id)sender
{
    UIImagePickerController *picker = [[UIImagePickerController alloc] init];
    picker.delegate = self;
    [self presentViewController:picker animated:YES completion:nil];
}
- (IBAction)onCameraClicked:(id)sender
{
    UIImagePickerController *picker = [[UIImagePickerController alloc] init];
    picker.sourceType = UIImagePickerControllerSourceTypeCamera;
    picker.delegate = self;
    [self presentViewController:picker animated:YES completion:nil];
}

- (void)didReceiveMemoryWarning
{
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
