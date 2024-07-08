#include <yarp/cv/Cv.h>
#include <yarp/os/Log.h>
#include <yarp/os/LogStream.h>
#include <yarpRTMPose.h>
#include "mmdeploy/detector.hpp"
#include "utils/visualize.h"
#include <fstream>
#include <cstdlib>

RTMPose::RTMPose(std::string det_model_path,
                 std::string pose_model_path,
                 std::string dataset,
                 std::string device) : detector{mmdeploy::Model(det_model_path), mmdeploy::Device(device)},
                                       pose_model{mmdeploy::Model(pose_model_path), mmdeploy::Device(device)},
                                       dataset{dataset},
                                       device{device},
                                       det_label{0},
                                       det_thr{0.5}
{
}

std::pair<std::vector<mmdeploy_rect_t>, mmdeploy::cxx::PoseDetector::Result> RTMPose::inference(const cv::Mat &img)
{
    mmdeploy::Detector::Result dets = detector.Apply(img);
    std::vector<mmdeploy_rect_t> bboxes;
    for (const auto &det : dets)
    {
        if (det.label_id == this->det_label && det.score > this->det_thr)
        {
            bboxes.push_back(det.bbox);
        }
    }

    // apply pose detector, if no bboxes are provided, full image will be used.
    mmdeploy::PoseDetector::Result poses = pose_model.Apply(img, bboxes);

    return std::make_pair(bboxes, poses);
}

cv::Mat RTMPose::paint(const cv::Mat &img,
                       const std::vector<mmdeploy_rect_t> &bboxes,
                       const mmdeploy::PoseDetector::Result poses)
{

    utils::Visualize v;

    // TODO: set skeleton can be done at config time
    if (dataset == "coco_wholebody")
    {
        v.set_skeleton(utils::Skeleton::get("coco-wholebody"));
    }
    else if (dataset == "coco")
    {
        v.set_skeleton(utils::Skeleton::get("coco"));
    }
    else
    {
        yWarning() << "Cannot paint the given dataset.";
        return cv::Mat();
    }

    auto sess = v.get_session(img);

    for (size_t i = 0; i < bboxes.size(); ++i)
    {
        sess.add_bbox(bboxes[i], -1, -1);
        sess.add_pose(poses[i].point, poses[i].score, poses[i].length, this->det_thr);
    }

    return sess.get().clone();
}

bool yarpRTMPose::configure(yarp::os::ResourceFinder &rf)
{
    if(rf.check("help"))
    {
        yInfo() << "RTMPose module. This module uses the RTMPOSE model to detect and estimate human poses in images.";
        yInfo() << "Setup: ";
        yInfo() << "Set the environment variable MMDEPLOY_DIR to the path where the models are stored.";
        yInfo() << "Usage: ";
        yInfo() << "--period: module period";
        yInfo() << "--det_model_path: path to the detection model. Default: MMDEPLOY_DIR/rtmpose-ort/rtmdet-nano";
        yInfo() << "--pose_model_path: path to the pose model. Default: MMDEPLOY_DIR/rtmpose-ort/rtmpose-l";
        yInfo() << "--dataset: dataset used for inference. Default: coco_wholebody.";
        yInfo() << "--device: device used for inference. Default: cuda";
        yInfo() << "--module_name: name of the module. Default: yarpRTMPose";
        yInfo() << "--openpose_format: if true, the output will be in openpose format";
        return false;
    }

    const char* mmdeploy_path_env = std::getenv("MMDEPLOY_DIR");
    std::string mmdeploy_path = mmdeploy_path_env ? std::string(mmdeploy_path_env) : "/mmdeploy";
    std::string det_model_default_path = mmdeploy_path + "/rtmpose-ort/rtmdet-nano";
    std::string pose_model_default_path = mmdeploy_path + "/rtmpose-ort/rtmpose-l";

    this->period = rf.check("period", yarp::os::Value(0.01)).asFloat32();
    this->det_model_path = rf.check("det_model_path", yarp::os::Value(det_model_default_path)).asString();
    this->pose_model_path = rf.check("pose_model_path", yarp::os::Value(pose_model_default_path)).asString();
    this->dataset = rf.check("dataset", yarp::os::Value("coco_wholebody")).asString();
    this->device = rf.check("device", yarp::os::Value("cuda")).asString();
    this->module_name = rf.check("module_name", yarp::os::Value("yarpRTMPose")).asString();
    this->openpose_format = rf.check("openpose_format", yarp::os::Value(false)).asBool();

    std::string dataset_file = rf.findFile(this->dataset + ".json");
    // Create the inference engine
    this->inferencer = std::make_unique<RTMPose>(this->det_model_path, this->pose_model_path, this->dataset, this->device);

    // Open yarp ports
    inPort.open("/" + module_name + "/image:i");
    outPort.open("/" + module_name + "/image:o");
    targetPort.open("/" + module_name + "/target:o");

    // Read dataset info json
    if (dataset_file == "")
    {
        yError() << "Dataset file was not found. Either dataset is invalid or the corresponding json file is not installed";
        return -1;
    }

    yDebug() << "Dataset file:" << dataset_file;
    std::ifstream r(dataset_file);
    keypointInfo = json::parse(r);

    if (openpose_format)
    {
        // Convert names to openpose standard
        std::string coco_to_op_filename = rf.findFile("coco_to_op.json");

        if (coco_to_op_filename == "")
        {
            yError() << "coco_to_op.json not found";
        }

        std::ifstream coco_to_op(coco_to_op_filename);
        auto coco_to_op_json = json::parse(coco_to_op);

        for (auto &[key, value] : coco_to_op_json.items())
        {

            for (auto &[rtm_key, rtm_value] : keypointInfo.items())
            {
                // If the original name is equal to the key of coco_to_op then we swap with OP name
                auto &orig_kp_name = rtm_value["name"];
                if (orig_kp_name == key)
                {
                    orig_kp_name = value;
                    break;
                }
            }
        }

        // Add two keypoints not present in COCO
        std::string op_not_in_coco_filename = rf.findFile("op_not_in_coco.json");
        if (op_not_in_coco_filename == "")
        {
            yError() << "op_not_in_coco.json not found";
        }
        std::ifstream op_not_in_coco(op_not_in_coco_filename);
        this->op_not_in_coco_json = json::parse(op_not_in_coco);
    }

    std::tie(face_keypoint_idx_start, face_keypoint_idx_end) = faceKeypointsIdxs();

    return true;
}

bool yarpRTMPose::updateModule()
{
    // mutex.wait();
    auto *frame = inPort.read();

    if (frame)
    {
        cv::Mat img = yarp::cv::toCvMat(*frame);
        auto [bboxes, keypoints] = inferencer->inference(img);

        // Convert keypoints to bottle format
        yarp::os::Bottle &target = targetPort.prepare();
        target.clear();
        target.addList().read(this->kpToBottle(keypoints));
        targetPort.write();

        cv::Mat keypoints_img = inferencer->paint(img, bboxes, keypoints);

        auto &outImage = outPort.prepare();
        outImage = yarp::cv::fromCvMat<yarp::sig::PixelRgb>(keypoints_img);
        // outImage = yarp::cv::fromCvMat<yarp::sig::PixelRgb>(img);
        outPort.write();
    }
    // mutex.post();

    return true;
}

bool yarpRTMPose::interruptModule()
{
    return close();
}

bool yarpRTMPose::close()
{
    this->inPort.close();
    this->outPort.close();
    this->targetPort.close();
    return true;
}

double yarpRTMPose::getPeriod()
{
    return period;
}

yarp::os::Bottle yarpRTMPose::kpToBottle(const mmdeploy::cxx::PoseDetector::Result &keypoints)
{
    yarp::os::Bottle targetBottle;

    if (openpose_format && dataset == "coco_wholebody")
    {

        // Standard format
        //(((kp x y score)...))
        // Openpose format
        //(((kp x y score)... (face (x y score) (x y score) ...) (kp x y score)))

        for (auto &skeleton : keypoints)
        {
            yarp::os::Bottle skeletonBottle;

            // Adding fake keypoint with 0 score to keep the format
            for (auto &[key, value] : this->op_not_in_coco_json.items())
            {
                yarp::os::Bottle keypointBottle;
                keypointBottle.addString(value);
                keypointBottle.addFloat32(0);
                keypointBottle.addFloat32(0);
                keypointBottle.addFloat32(0);

                skeletonBottle.addList().read(keypointBottle);
            }

            for (size_t i = 0; i < face_keypoint_idx_start; ++i)
            {
                yarp::os::Bottle keypointBottle;

                const auto &kp = skeleton.point[i];
                keypointBottle.addString(this->keypointInfo[std::to_string(i)]["name"]);
                keypointBottle.addFloat32(kp.x);
                keypointBottle.addFloat32(kp.y);
                keypointBottle.addFloat32(skeleton.score[i]);

                skeletonBottle.addList().read(keypointBottle);
            }

            yarp::os::Bottle faceBottle;
            faceBottle.addString("Face");

            for (size_t i = face_keypoint_idx_start; i < face_keypoint_idx_end; ++i)
            {
                yarp::os::Bottle faceKeypointBottle;
                const auto &kp = skeleton.point[i];
                faceKeypointBottle.addFloat32(kp.x);
                faceKeypointBottle.addFloat32(kp.y);
                faceKeypointBottle.addFloat32(skeleton.score[i]);

                faceBottle.addList().read(faceKeypointBottle);
            }

            skeletonBottle.addList().read(faceBottle);

            for (size_t i = face_keypoint_idx_end; i < skeleton.length; ++i)
            {
                yarp::os::Bottle keypointBottle;

                const auto &kp = skeleton.point[i];
                keypointBottle.addString(this->keypointInfo[std::to_string(i)]["name"]);
                keypointBottle.addFloat32(kp.x);
                keypointBottle.addFloat32(kp.y);
                keypointBottle.addFloat32(skeleton.score[i]);

                skeletonBottle.addList().read(keypointBottle);
            }

            targetBottle.addList().read(skeletonBottle);
        }
    }
    else
    {
        for (auto &skeleton : keypoints)
        {
            yarp::os::Bottle skeletonBottle;

            for (size_t i = 0; i < skeleton.length; ++i)
            {
                yarp::os::Bottle keypointBottle;

                const auto &kp = skeleton.point[i];
                keypointBottle.addString(this->keypointInfo[std::to_string(i)]["name"]);
                keypointBottle.addFloat32(kp.x);
                keypointBottle.addFloat32(kp.y);

                skeletonBottle.addList().read(keypointBottle);
            }

            targetBottle.addList().read(skeletonBottle);
        }
    }

    return targetBottle;
}

std::pair<size_t, size_t> yarpRTMPose::faceKeypointsIdxs()
{

    // TODO: make it more general like the (bugged) version below
    // std::string::size_type n;
    // size_t start = 0;
    // size_t end = 0;

    // json::iterator it = keypointInfo.begin();

    // while(start == 0 && it != keypointInfo.end())
    // {
    //     std::string value = it.value()["name"].get<std::string>();
    //     n = value.find("face");
    //     if(std::string::npos != n) //The first time we see "face" as a keypoint we stop this half
    //     {
    //         start = std::stoi(it.key());
    //     }
    //     ++it;
    // }

    // while(end == 0 && it != keypointInfo.end())
    // {
    //     std::string value = it.value()["name"].get<std::string>();
    //     n = value.find("face");
    //     if(std::string::npos == n) //The first time we don't see "face" as a keypoint we stop
    //     {
    //         end = std::stoi(it.key());
    //     }
    //     ++it;
    // }

    std::size_t start = 23;
    std::size_t end = 91;

    return std::make_pair(start, end);
}
