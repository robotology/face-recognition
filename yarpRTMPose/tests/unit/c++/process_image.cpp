#include <string>
#include <yarpRTMPose.h>
#include <filesystem>
#include <iostream>
#include <catch2/catch_test_macros.hpp>
#include <yarp/os/Log.h>
#include <yarp/os/LogStream.h>

TEST_CASE("Keypoints from image","[yarpRTMPose]")
{
    const char* mmdeploy_path_env = std::getenv("MMDEPLOY_DIR");

    if(!mmdeploy_path_env)
    {
        yError() << "Please set MMDEPLOY_DIR environment variable";
        return;
    }

    std::string det_model_path = std::string(mmdeploy_path_env) + "/rtmpose-ort/rtmdet-nano";
    std::string pose_model_path = std::string(mmdeploy_path_env) + "/rtmpose-ort/rtmpose-l";
    
    const std::filesystem::path img_path{"../../tests/data/000000000785.jpg"};

    cv::Mat test_img{cv::imread(img_path)};

    if(test_img.empty())
    {
        std::cout << "Could not read the image" << img_path << std::endl;
    }

    const std::string dataset = "coco_wholebody"; 
    const std::string device = "cuda";

    RTMPose inferencer{det_model_path,pose_model_path,dataset,device};
    auto [bboxes, keypoints] = inferencer.inference(test_img);

    // There is 1 person in the image
    REQUIRE(keypoints.size() == bboxes.size());

    //COCO_wholebody has 133 kp
    REQUIRE(133 == keypoints[0].length);

} 

TEST_CASE("RTMPose pipeline","[yarpRTMPose]")
{
    const char* mmdeploy_path_env = std::getenv("MMDEPLOY_DIR");

    if(!mmdeploy_path_env)
    {
        yError() << "Please set MMDEPLOY_DIR environment variable";
        return;
    }

    std::string det_model_path = std::string(mmdeploy_path_env) + "/rtmpose-ort/rtmdet-nano";
    std::string pose_model_path = std::string(mmdeploy_path_env) + "/rtmpose-ort/rtmpose-l";

    // 1. Build keypoints vector
    const std::filesystem::path img_path{"../../tests/data/000000000785.jpg"};
    cv::Mat test_img = cv::imread(img_path);

    if(test_img.empty())
    {
        std::cout << "Could not read the image" << img_path;
    }

    const std::string dataset = "coco_wholebody"; 
    const std::string device = "cuda";

    RTMPose inferencer{det_model_path,pose_model_path,dataset,device};
    auto [bboxes, keypoints] = inferencer.inference(test_img);

    // 2. paint the results
    cv::Mat kp_img = inferencer.paint(test_img,bboxes,keypoints);

    // 3. check results
    cv::imwrite("/test.png",kp_img);
}