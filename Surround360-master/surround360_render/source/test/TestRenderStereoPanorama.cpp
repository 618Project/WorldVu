/**
* Copyright (c) 2016-present, Facebook, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-style license found in the
* LICENSE_render file in the root directory of this subproject. An additional grant
* of patent rights can be found in the PATENTS file in the same directory.
*/

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "Camera.h"
#include "CameraMetadata.h"
#include "ColorAdjustmentSampleLogger.h"
#include "CvUtil.h"
#include "Filter.h"
#include "ImageWarper.h"
#include "IntrinsicCalibration.h"
#include "MathUtil.h"
#include "MonotonicTable.h"
#include "NovelView.h"
#include "OpticalFlowFactory.h"
#include "OpticalFlowVisualization.h"
#include "PoleRemoval.h"
#include "SideCameraBrightnessAdjustment.h"
#include "StringUtil.h"
#include "SystemUtil.h"
#include "VrCamException.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <ctime>
#include "Sharpen.h"
// NOTE: BY SCHANDA
// #include "checkpoint.h"
#include "defines.h"

using namespace cv;
using namespace std;
using namespace surround360;
using namespace surround360::calibration;
using namespace surround360::math_util;
using namespace surround360::optical_flow;
using namespace surround360::util;
using namespace surround360::warper;
using namespace surround360::color_adjust;

DEFINE_string(src_intrinsic_param_file,   "",             "path to read intrinsic matrices");
DEFINE_string(rig_json_file,              "",             "path to json file drescribing camera array");
DEFINE_string(ring_rectify_file,          "NONE",         "path to rectification transforms file for ring of cameras");
DEFINE_string(imgs_dir,                   "",             "path to folder of images with names matching cameras in the rig file");
DEFINE_string(output_data_dir,            "",             "path to write spherical projections for debugging");
DEFINE_string(prev_frame_data_dir,        "NONE",         "path to data for previous frame; used for temporal regularization");
DEFINE_string(output_cubemap_path,        "",             "path to write output oculus 360 cubemap");
DEFINE_string(output_equirect_path,       "",             "path to write output oculus 360 cubemap");
DEFINE_double(interpupilary_dist,         6.4,            "separation of eyes for stereo, spherical_in whatever units the rig json uses.");
DEFINE_int32(side_alpha_feather_size,     100,            "alpha feather for projection of side cameras to spherical coordinates");
DEFINE_int32(std_alpha_feather_size,      31,             "alpha feather for all other purposes. must be odd");
DEFINE_bool(save_debug_images,            false,          "if true, lots of debug images are generated");
DEFINE_double(sharpenning,                0.0f,           "0.0 to 1.0 amount of sharpenning");
DEFINE_bool(enable_top,                   false,          "is there a top camera?");
DEFINE_bool(enable_bottom,                false,          "are there two bottom cameras?");
DEFINE_bool(enable_pole_removal,          false,          "if true, pole removal masks are used; if false, primary bottom camera is used");
DEFINE_string(bottom_pole_masks_dir,      "",             "path to bottom camera pole masks dir");
DEFINE_string(side_flow_alg,              "pixflow_low",  "which optical flow algorithm to use for sides");
DEFINE_string(polar_flow_alg,             "pixflow_low",  "which optical flow algorithm to use for top/bottom warp with sides");
DEFINE_string(poleremoval_flow_alg,       "pixflow_low",  "which optical flow algorithm to use for pole removal with secondary bottom camera");
DEFINE_double(zero_parallax_dist,         10000.0,        "distance where parallax is zero");
DEFINE_int32(eqr_width,                   256,            "width of spherical projection image (0 to 2pi)");
DEFINE_int32(eqr_height,                  128,            "height of spherical projection image (0 to pi)");
DEFINE_int32(final_eqr_width,             3480,           "resize before stacking stereo equirect width");
DEFINE_int32(final_eqr_height,            960,            "resize before stacking stereo equirect height");
DEFINE_int32(cubemap_width,               1536,           "face width of output cubemaps");
DEFINE_int32(cubemap_height,              1536,           "face height of output cubemaps");
DEFINE_string(cubemap_format,             "video",        "either video or photo");
DEFINE_string(brightness_adjustment_dest, "",             "if non-empty, a brightness adjustment file will be written to this path");
DEFINE_string(brightness_adjustment_src,  "",             "if non-empty, a brightness level adjustment file will be read from this path");
DEFINE_bool(enable_render_coloradjust,    false,          "if true, color and brightness of images will be automatically adjusted to make smoother blends (in the renderer, not in the ISP step)");
DEFINE_bool(new_rig_format,               false,          "use new rig and camera json format");

const Camera::Vector3 kGlobalUp = Camera::Vector3::UnitZ();

  // __global__ void temporary () {
  //   int a;
  //   a = 3*4;
  //   printf ("a: %d\n", a);
  // }
  
Scalar getMSSIM( const Mat& i1, const Mat& i2)
{
 const double C1 = 6.5025, C2 = 58.5225;
 /***************************** INITS **********************************/
 int d     = CV_32F;

 Mat I1, I2;
 i1.convertTo(I1, d);           // cannot calculate on one byte large values
 i2.convertTo(I2, d);

 Mat I2_2   = I2.mul(I2);        // I2^2
 Mat I1_2   = I1.mul(I1);        // I1^2
 Mat I1_I2  = I1.mul(I2);        // I1 * I2

 /***********************PRELIMINARY COMPUTING ******************************/

 Mat mu1, mu2;   //
 GaussianBlur(I1, mu1, Size(11, 11), 1.5);
 GaussianBlur(I2, mu2, Size(11, 11), 1.5);

 Mat mu1_2   =   mu1.mul(mu1);
 Mat mu2_2   =   mu2.mul(mu2);
 Mat mu1_mu2 =   mu1.mul(mu2);

 Mat sigma1_2, sigma2_2, sigma12;

 GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
 sigma1_2 -= mu1_2;

 GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
 sigma2_2 -= mu2_2;

 GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
 sigma12 -= mu1_mu2;

 ///////////////////////////////// FORMULA ////////////////////////////////
 Mat t1, t2, t3;

 t1 = 2 * mu1_mu2 + C1;
 t2 = 2 * sigma12 + C2;
 t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

 t1 = mu1_2 + mu2_2 + C1;
 t2 = sigma1_2 + sigma2_2 + C2;
 t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

 Mat ssim_map;
 divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

 Scalar mssim = mean( ssim_map ); // mssim = average of ssim map
 return mssim;
}
  
// represents either new or old rig format depending on FLAGS_new_rig_format
struct RigDescription {
  // new format fields
  Camera::Rig rig;
  Camera::Rig rigSideOnly;

  // old format fields
  float cameraRingRadius;
  vector<CameraMetadata> camModelArrayWithTop;
  vector<CameraMetadata> camModelArray;
  vector<Mat> sideCamTransforms;

  RigDescription(const string& filename, bool useNewFormat);

  bool isNewFormat() const { return !rig.empty(); }

  // find the camera that is closest to pointing in the provided direction
  // ignore those with excessive distance from the camera axis to the rig center
  const Camera& findCameraByDirection(
      const Camera::Vector3& direction,
      const Camera::Real distCamAxisToRigCenterMax = 1.0) const {
    CHECK(isNewFormat());
    const Camera* best = nullptr;
    for (const Camera& camera : rig) {
      if (best == nullptr ||
          best->forward().dot(direction) < camera.forward().dot(direction)) {
        if (distCamAxisToRigCenter(camera) <= distCamAxisToRigCenterMax) {
          best = &camera;
        }
      }
    }
    return *CHECK_NOTNULL(best);
  }

  // find the camera with the largest distance from camera axis to rig center
  const Camera& findLargestDistCamAxisToRigCenter() const {
    CHECK(isNewFormat());
    const Camera* best = &rig.back();
    for (const Camera& camera : rig) {
      if (distCamAxisToRigCenter(camera) > distCamAxisToRigCenter(*best)) {
        best = &camera;
      }
    }
    return *best;
  }

  string getTopCameraId() const {
    return isNewFormat()
      ? findCameraByDirection(kGlobalUp).id
      : getTopCamModel(camModelArrayWithTop).cameraId;
  }

  string getBottomCameraId() const {
    return isNewFormat()
      ? findCameraByDirection(-kGlobalUp).id
      : getBottomCamModel(camModelArrayWithTop).cameraId;
  }

  string getBottomCamera2Id() const {
    return isNewFormat()
      ? findLargestDistCamAxisToRigCenter().id
      : getBottomCamModel2(camModelArrayWithTop).cameraId;
  }

  int getSideCameraCount() const {
    return isNewFormat() ? rigSideOnly.size() : camModelArray.size();
  }

  string getSideCameraId(const int idx) const {
    return isNewFormat() ? rigSideOnly[idx].id : camModelArray[idx].cameraId;
  }

  float getRingRadius() const {
    return isNewFormat() ? rigSideOnly[0].position.norm() : cameraRingRadius;
  }

  vector<Mat> loadSideCameraImages(const string& imageDir) const {
    const string extension = getImageFileExtension(imageDir);

    VLOG(1) << "loadSideCameraImages spawning threads";
    vector<std::thread> threads;
    vector<Mat> images(getSideCameraCount());
    for (int i = 0; i < getSideCameraCount(); ++i) {
      const string imageFilename = getSideCameraId(i) + "." + extension;
      const string imagePath = imageDir + "/" + imageFilename;
      VLOG(1) << "imagePath = " << imagePath;
      threads.emplace_back(
        imreadInStdThread,
        imagePath,
        CV_LOAD_IMAGE_COLOR,
        &(images[i])
      );
    }
    for (auto& thread : threads) {
      thread.join();
    }

    return images;
  }

private:
  static Camera::Real distCamAxisToRigCenter(const Camera& camera) {
    return camera.rig(camera.principal).distance(Camera::Vector3::Zero());
  }
};

RigDescription::RigDescription(const string& filename, const bool useNewFormat) {
  if (useNewFormat) {
    rig = Camera::loadRig(filename);
    for (const Camera& camera : rig) {
      if (camera.group.find("side") != string::npos) {
        rigSideOnly.emplace_back(camera);
      }
    }
  } else {
    requireArg(FLAGS_src_intrinsic_param_file, "src_intrinsic_param_file");
    // load camera meta data and source images
    VLOG(1) << "Reading camera model json";
    camModelArrayWithTop =
      readCameraProjectionModelArrayFromJSON(
        filename,
        cameraRingRadius);

    VLOG(1) << "Verifying image filenames";
    verifyImageDirFilenamesMatchCameraArray(camModelArrayWithTop, FLAGS_imgs_dir);

    VLOG(1) << "Removing top and bottom cameras";
    camModelArray =
      removeTopAndBottomFromCamArray(camModelArrayWithTop);

    // read bundle adjustment for side cameras
    if (FLAGS_ring_rectify_file == "NONE") {
      LOG(WARNING) << "No ring rectification file specified";
      for (int i = 0; i < camModelArray.size(); ++i) {
        sideCamTransforms.push_back(Mat());
      }
    } else {
      VLOG(1) << "Reading ring rectification file: " << FLAGS_ring_rectify_file;
      FileStorage fileStorage(FLAGS_ring_rectify_file, FileStorage::READ);
      if (!fileStorage.isOpened()) {
        throw VrCamException("file read failed: " + FLAGS_ring_rectify_file);
      }

      for (int i = 0; i < camModelArray.size(); ++i) {
        Mat transformForCamI;
        fileStorage[camModelArray[i].cameraId] >> transformForCamI;
        sideCamTransforms.push_back(transformForCamI);
      }
    }
  }

  // validation
  CHECK_EQ(useNewFormat, isNewFormat());
  CHECK_NE(getSideCameraCount(), 0);

  if (FLAGS_eqr_width % getSideCameraCount() != 0) {
    VLOG(1) << "Number of side cameras:" << getSideCameraCount();
    VLOG(1) << "Suggested widths:";
    for (int i = FLAGS_eqr_width * 0.9; i < FLAGS_eqr_width * 1.1; ++i) {
      if (i % getSideCameraCount() == 0) {
        VLOG(1) << i;
      }
    }
    throw VrCamException("eqr_width must be evenly divisible by the number of cameras");
  }

}

// sample the camera's fov cone to find the closest point to the image center
float approximateUsablePixelsRadius(const Camera& camera) {
  const Camera::Real fov = camera.getFov();
  const Camera::Real kStep = 2 * M_PI / 10.0;
  Camera::Real result = camera.resolution.norm();
  for (Camera::Real a = 0; a < 2 * M_PI; a += kStep) {
    Camera::Vector3 ortho = cos(a) * camera.right() + sin(a) * camera.up();
    Camera::Vector3 direction = cos(fov) * camera.forward() + sin(fov) * ortho;
    Camera::Vector2 pixel = camera.pixel(camera.position + direction);
    result = min(result, (pixel - camera.resolution / 2.0).norm());
  }
  return result;
}

// measured in radians from forward
float approximateFov(const Camera& camera, const bool vertical) {
  Camera::Vector2 a = camera.principal;
  Camera::Vector2 b = camera.principal;
  if (vertical) {
    a.y() = 0;
    b.y() = camera.resolution.y();
  } else {
    a.x() = 0;
    b.x() = camera.resolution.x();
  }
  return acos(max(
    camera.rig(a).direction().dot(camera.forward()),
    camera.rig(b).direction().dot(camera.forward())));
}

// measured in radians from forward
float approximateFov(const Camera::Rig& rig, const bool vertical) {
  float result = 0;
  for (const auto& camera : rig) {
    result = std::max(result, approximateFov(camera, vertical));
  }
  return result;
}

// project the image of a single camera into spherical coordinates
void projectCamImageToSphericalThread(
    float brightnessAdjustment,
    Mat* intrinsic,
    Mat* distCoeffs,
    const CameraMetadata* cam,
    const Mat* perspectiveTransform,
    Mat* camImage,
    Mat* outProjectedImage) {

  Mat projectedImage;
  if (cam->isFisheye) {
    VLOG(1) << "Projecting fisheye camera";
    projectedImage = sideFisheyeToSpherical(
      *camImage,
      *cam,
      FLAGS_eqr_width * (cam->fovHorizontal / 360.0),
      FLAGS_eqr_height * ((cam->fovHorizontal / cam->aspectRatioWH) / 180.0));
  } else {
    VLOG(1) << "Projecting non-fisheye camera";
    const bool skipUndistort = (FLAGS_src_intrinsic_param_file == "NONE");
    projectedImage = undistortToSpherical(
      cam->fovHorizontal,
      cam->fovHorizontal / cam->aspectRatioWH,
      FLAGS_eqr_width * (cam->fovHorizontal / 360.0),
      FLAGS_eqr_height * ((cam->fovHorizontal / cam->aspectRatioWH) / 180.0),
      *intrinsic,
      *distCoeffs,
      *perspectiveTransform,
      *camImage,
      FLAGS_side_alpha_feather_size,
      skipUndistort);
  }

  // if we got a non-zero brightness adjustment, apply it
  if (FLAGS_enable_render_coloradjust && brightnessAdjustment != 0.0f) {
    projectedImage = addBrightnessAndClamp(
      projectedImage, brightnessAdjustment);
  }

  *outProjectedImage = projectedImage;
}

void projectSideToSpherical(
    Mat& dst,
    const Mat& src,
    const Camera& camera,
    const float leftAngle,
    const float rightAngle,
    const float topAngle,
    const float bottomAngle,
    const float brightnessAdjust) {
  // convert, clone or reference, as needed
  Mat tmp = src;
  if (src.channels() == 3) {
    cvtColor(src, tmp, CV_BGR2BGRA);
  } else if (FLAGS_side_alpha_feather_size) {
    tmp = src.clone();
  }
  // feather
  if (FLAGS_side_alpha_feather_size) {
    for (int y = 0; y < FLAGS_side_alpha_feather_size; ++y) {
      const uint8_t alpha =
        255.0f * float(y + 0.5f) / float(FLAGS_side_alpha_feather_size);
      for (int x = 0; x < tmp.cols; ++x) {
        tmp.at<Vec4b>(y, x)[3] = alpha;
        tmp.at<Vec4b>(tmp.rows - 1 - y, x)[3] = alpha;
      }
    }
  }
  // remap
  bicubicRemapToSpherical(
    dst,
    tmp,
    camera,
    leftAngle,
    rightAngle,
    topAngle,
    bottomAngle);
  // adjust brightness
  if (FLAGS_enable_render_coloradjust && brightnessAdjust != 0.0f) {
    dst += Scalar(brightnessAdjust, brightnessAdjust, brightnessAdjust, 0);
  }
}

// project all of the (side) cameras' images into spherical coordinates
void projectSphericalCamImages(
      const RigDescription& rig,
      const string& imagesDir,
      vector<Mat>& projectionImages) {

  VLOG(1) << "Projecting side camera images to spherical coordinates";

  const double startLoadCameraImagesTime = getCurrTimeSec();
  vector<Mat> camImages = rig.loadSideCameraImages(imagesDir);
  const double endLoadCameraImagesTime = getCurrTimeSec();
  VLOG(1) << "Time to load images from file: "
    << endLoadCameraImagesTime - startLoadCameraImagesTime
    << " sec";

  // if we got intrinsic lens parameters, read them to correct distortion
  Mat intrinsic, distCoeffs;
  if (FLAGS_src_intrinsic_param_file == "NONE") {
    VLOG(1) << "src_intrinsic_param_file = NONE. no intrinsics loaded";
  } else {
    FileStorage fileStorage(FLAGS_src_intrinsic_param_file, FileStorage::READ);
    if (fileStorage.isOpened()) {
      fileStorage["intrinsic"] >> intrinsic;
      fileStorage["distCoeffs"] >> distCoeffs;
    } else {
      throw VrCamException("file read failed: " + FLAGS_src_intrinsic_param_file);
    }
  }

  // if we got a brightness adjustments file, read the values
  vector<float> brightnessAdjustments(rig.getSideCameraCount(), 0.0f);
  if (FLAGS_enable_render_coloradjust &&
      !FLAGS_brightness_adjustment_src.empty()){
    LOG(INFO) << "reading brightness adjustment file: "
      << FLAGS_brightness_adjustment_src;
    ifstream brightnessAdjustFile(FLAGS_brightness_adjustment_src);
    if (!brightnessAdjustFile) {
      throw VrCamException(
        "file read failed: " + FLAGS_brightness_adjustment_src);
    }
    const string strData(
      (istreambuf_iterator<char>(brightnessAdjustFile)),
      istreambuf_iterator<char>());
    vector<string> brightnessStrs = stringSplit(strData , ',');
    if (brightnessStrs.size() != rig.getSideCameraCount()) {
      throw VrCamException(
        "expected number of brightness adjustment values to match number of "
        "side cameras. got # cameras = " + to_string(rig.getSideCameraCount()) +
        " and # brightness adjustments = " +
        to_string(brightnessAdjustments.size()));
    }
    for (int i = 0; i < brightnessStrs.size(); ++i) {
      brightnessAdjustments[i] = std::stof(brightnessStrs[i]);
    }
  }

  projectionImages.resize(camImages.size());
  vector<std::thread> threads;
  if (rig.isNewFormat()) {
    const float hRadians = 2 * approximateFov(rig.rigSideOnly, false);
    const float vRadians = 2 * approximateFov(rig.rigSideOnly, true);
    for (int camIdx = 0; camIdx < camImages.size(); ++camIdx) {
      const Camera& camera = rig.rigSideOnly[camIdx];
      projectionImages[camIdx].create(
        FLAGS_eqr_height * vRadians / M_PI,
        FLAGS_eqr_width * hRadians / (2 * M_PI),
        CV_8UC4);
      // the negative sign here is so the camera array goes clockwise
      float direction = -float(camIdx) / float(camImages.size()) * 2.0f * M_PI;
      threads.emplace_back(
        projectSideToSpherical,
        ref(projectionImages[camIdx]),
        cref(camImages[camIdx]),
        cref(camera),
        direction + hRadians / 2,
        direction - hRadians / 2,
        vRadians / 2,
        -vRadians / 2,
        brightnessAdjustments[camIdx]);
    }
  } else {
    for (int camIdx = 0; camIdx < camImages.size(); ++camIdx) {
      threads.emplace_back(
        projectCamImageToSphericalThread,
        brightnessAdjustments[camIdx],
        &intrinsic,
        &distCoeffs,
        &rig.camModelArray[camIdx],
        &rig.sideCamTransforms[camIdx],
        &camImages[camIdx],
        &projectionImages[camIdx]
      );
    }
  }
  for (std::thread& t : threads) { t.join(); }

  if (FLAGS_save_debug_images) {
    for (int camIdx = 0; camIdx < rig.getSideCameraCount(); ++camIdx) {
      const string cropImageFilename = FLAGS_output_data_dir +
        "/projections/crop_" + rig.getSideCameraId(camIdx) + ".png";
      imwriteExceptionOnFail(cropImageFilename, projectionImages[camIdx]);
    }
  }
}

// this is where the main work of optical flow for adjacent side cameras is done
void prepareNovelViewGeneratorThread(
    const int overlapImageWidth,
    const int leftIdx, // only used to determine debug image filename
    Mat* imageL,
    Mat* imageR,
    NovelViewGenerator* novelViewGen,
    int mode) {

  // time_checkpoint ("");
  Mat overlapImageL = (*imageL)(Rect(
    imageL->cols - overlapImageWidth, 0, overlapImageWidth, imageL->rows));
  Mat overlapImageR = (*imageR)(Rect(0, 0, overlapImageWidth, imageR->rows));

  // save the images that are going into flow. we will need them in the next frame
  imwriteExceptionOnFail(
    FLAGS_output_data_dir + "/flow_images/overlap_" + std::to_string(leftIdx) + "_L.png",
    overlapImageL);
  imwriteExceptionOnFail(
    FLAGS_output_data_dir + "/flow_images/overlap_" + std::to_string(leftIdx) + "_R.png",
    overlapImageR);
  // time_checkpoint("first");

  // read the previous frame's flow results, if available
  Mat prevFrameFlowLtoR, prevFrameFlowRtoL, prevOverlapImageL, prevOverlapImageR;
  if (FLAGS_prev_frame_data_dir != "NONE") {
    VLOG(1) << "Reading previous frame flow and images from: "
      << FLAGS_prev_frame_data_dir;
    prevFrameFlowLtoR = readFlowFromFile(
      FLAGS_prev_frame_data_dir + "/flow/flowLtoR_" + std::to_string(leftIdx) + ".bin");
    prevFrameFlowRtoL = readFlowFromFile(
      FLAGS_prev_frame_data_dir + "/flow/flowRtoL_" + std::to_string(leftIdx) + ".bin");
    prevOverlapImageL = imreadExceptionOnFail(
      FLAGS_prev_frame_data_dir + "/flow_images/overlap_" + std::to_string(leftIdx) + "_L.png",
      -1);
    prevOverlapImageR = imreadExceptionOnFail(
      FLAGS_prev_frame_data_dir + "/flow_images/overlap_" + std::to_string(leftIdx) + "_R.png",
      -1);
    VLOG(1) << "Loaded previous frame's flow OK";
  }
  // time_checkpoint("second");
  // ~11/12 seconds hotspot

  // this is the call to actually compute optical flow
  novelViewGen->prepare(
    overlapImageL,
    overlapImageR,
    prevFrameFlowLtoR,
    prevFrameFlowRtoL,
    prevOverlapImageL,
    prevOverlapImageR,
    mode);
  // time_checkpoint("third");

  // get the results of flow and save them. we will need these for temporal regularization
  const Mat flowLtoR = novelViewGen->getFlowLtoR();
  const Mat flowRtoL = novelViewGen->getFlowRtoL();
  saveFlowToFile(
    flowLtoR,
    FLAGS_output_data_dir + "/flow/flowLtoR_" + std::to_string(leftIdx) + ".bin");
  saveFlowToFile(
    flowRtoL,
    FLAGS_output_data_dir + "/flow/flowRtoL_" + std::to_string(leftIdx) + ".bin");
  // time_checkpoint("fourth");
}

// a "chunk" is the portion from a pair of overlapping cameras. returns left/right images
void renderStereoPanoramaChunksThread(
    const int leftIdx, // left camera
    const int numCams,
    const int camImageWidth,
    const int camImageHeight,
    const int numNovelViews,
    const float fovHorizontalRadians,
    const float vergeAtInfinitySlabDisplacement,
    NovelViewGenerator* novelViewGen,
    Mat* chunkL,
    Mat* chunkR) {

  int currChunkX = 0; // current column in chunk to write
  LazyNovelViewBuffer lazyNovelViewBuffer(FLAGS_eqr_width / numCams, camImageHeight);
  for (int nvIdx = 0; nvIdx < numNovelViews; ++nvIdx) {
    const float shift = float(nvIdx) / float(numNovelViews);
    const float slabShift =
      float(camImageWidth) * 0.5f - float(numNovelViews - nvIdx);

    for (int v = 0; v < camImageHeight; ++v) {
      lazyNovelViewBuffer.warpL[currChunkX][v] =
        Point3f(slabShift + vergeAtInfinitySlabDisplacement, v, shift);
      lazyNovelViewBuffer.warpR[currChunkX][v] =
        Point3f(slabShift - vergeAtInfinitySlabDisplacement, v, shift);
    }
    ++currChunkX;
  }

  const int rightIdx = (leftIdx + 1) % numCams;
  pair<Mat, Mat> lazyNovelChunksLR = novelViewGen->combineLazyNovelViews(
    lazyNovelViewBuffer,
    leftIdx,
    rightIdx);
  *chunkL = lazyNovelChunksLR.first;
  *chunkR = lazyNovelChunksLR.second;
}

void st_generateRingOfNovelViewsAndRenderStereoSpherical (
    const float cameraRingRadius,
    const float camFovHorizontalDegrees,
    vector<Mat>& projectionImages,  // I/P
    Mat& panoImageL,   // O/P
    Mat& panoImageR,   // O/P
    double& opticalFlowRuntime,
    double& novelViewRuntime) {

  int mode = GPU_MODE;
  const int numCams = projectionImages.size();

  // this is the amount of horizontal overlap the cameras would have if they
  // were all perfectly aligned (in fact due to misalignment they overlap by a
  // different amount for each pair, but we ignore that to make it simple)
  const float fovHorizontalRadians = toRadians(camFovHorizontalDegrees);
  const float overlapAngleDegrees =
    (camFovHorizontalDegrees * float(numCams) - 360.0) / float(numCams);
  const int camImageWidth = projectionImages[0].cols;
  const int camImageHeight = projectionImages[0].rows;
  const int overlapImageWidth =
    float(camImageWidth) * (overlapAngleDegrees / camFovHorizontalDegrees);
  const int numNovelViews = camImageWidth - overlapImageWidth; // per image pair

  // setup parallel optical flow
  double startOpticalFlowTime = getCurrTimeSec();
  vector<NovelViewGenerator*> novelViewGenerators(projectionImages.size());
  vector<std::thread> threads;

  // time_checkpoint("");
  for (int leftIdx=0; leftIdx < projectionImages.size(); ++leftIdx) {
    const int rightIdx = (leftIdx+1) % projectionImages.size();
    novelViewGenerators[leftIdx] =
      new NovelViewGeneratorAsymmetricFlow(FLAGS_side_flow_alg);
//    prepareNovelViewGeneratorThread(
      threads.push_back(std::thread(
      prepareNovelViewGeneratorThread,
      overlapImageWidth,
      leftIdx,
      &projectionImages[leftIdx],
      &projectionImages[rightIdx],
      novelViewGenerators[leftIdx],
      mode));  // In which mode to run
  }
  for (std::thread& t : threads) { t.join(); }
  // time_checkpoint("first");

  opticalFlowRuntime = getCurrTimeSec() - startOpticalFlowTime;

  // lightfield/parallax formulas
  const float v =
    atanf(FLAGS_zero_parallax_dist / (FLAGS_interpupilary_dist / 2.0f));
  const float psi =
    asinf(sinf(v) * (FLAGS_interpupilary_dist / 2.0f) / cameraRingRadius);
  const float vergeAtInfinitySlabDisplacement =
    psi * (float(camImageWidth) / fovHorizontalRadians);
  const float theta = -M_PI / 2.0f + v + psi;
  const float zeroParallaxNovelViewShiftPixels =
    float(FLAGS_eqr_width) * (theta / (2.0f * M_PI));
  // time_checkpoint("second");
  // ~4.9/23 seconds

  double startNovelViewTime = getCurrTimeSec();
  // a "chunk" will be just the part of the panorama formed from one pair of
  // adjacent cameras. we will stack them horizontally to build the full
  // panorama. we do this so it can be parallelized.
  vector<Mat> panoChunksL(projectionImages.size(), Mat());
  vector<Mat> panoChunksR(projectionImages.size(), Mat());
  for (int leftIdx = 0; leftIdx < projectionImages.size(); ++leftIdx) {
    renderStereoPanoramaChunksThread(
      leftIdx,
      numCams,
      camImageWidth,
      camImageHeight,
      numNovelViews,
      fovHorizontalRadians,
      vergeAtInfinitySlabDisplacement,
      novelViewGenerators[leftIdx],
      &panoChunksL[leftIdx],
      &panoChunksR[leftIdx]
    );
  }
  // time_checkpoint("third");

  novelViewRuntime = getCurrTimeSec() - startNovelViewTime;

  // time_checkpoint("fourth");
  for (int leftIdx = 0; leftIdx < projectionImages.size(); ++leftIdx) {
    delete novelViewGenerators[leftIdx];
  }
  // time_checkpoint("fifth");

  panoImageL = stackHorizontal(panoChunksL);
  panoImageR = stackHorizontal(panoChunksR);

  panoImageL = offsetHorizontalWrap(panoImageL, zeroParallaxNovelViewShiftPixels);
  panoImageR = offsetHorizontalWrap(panoImageR, -zeroParallaxNovelViewShiftPixels);
  // time_checkpoint("sixth");
  
}


// generates a left/right eye equirect panorama using slices of novel views
void generateRingOfNovelViewsAndRenderStereoSpherical(
    const float cameraRingRadius,
    const float camFovHorizontalDegrees,
    vector<Mat>& projectionImages,
    Mat& panoImageL,
    Mat& panoImageR,
    double& opticalFlowRuntime,
    double& novelViewRuntime) {

  int mode = CPU_MODE;
  const int numCams = projectionImages.size();

  // this is the amount of horizontal overlap the cameras would have if they
  // were all perfectly aligned (in fact due to misalignment they overlap by a
  // different amount for each pair, but we ignore that to make it simple)
  const float fovHorizontalRadians = toRadians(camFovHorizontalDegrees);
  const float overlapAngleDegrees =
    (camFovHorizontalDegrees * float(numCams) - 360.0) / float(numCams);
  const int camImageWidth = projectionImages[0].cols;
  const int camImageHeight = projectionImages[0].rows;
  const int overlapImageWidth =
    float(camImageWidth) * (overlapAngleDegrees / camFovHorizontalDegrees);
  const int numNovelViews = camImageWidth - overlapImageWidth; // per image pair

  // setup parallel optical flow
  double startOpticalFlowTime = getCurrTimeSec();
  vector<NovelViewGenerator*> novelViewGenerators(projectionImages.size());
  vector<std::thread> threads;

  // time_checkpoint("");
  // ~14.8/23 seconds of hotspot
  for (int leftIdx = 0; leftIdx < projectionImages.size(); ++leftIdx) {
    const int rightIdx = (leftIdx + 1) % projectionImages.size();
    novelViewGenerators[leftIdx] =
      new NovelViewGeneratorAsymmetricFlow(FLAGS_side_flow_alg);
    threads.push_back(std::thread(
      prepareNovelViewGeneratorThread,
      overlapImageWidth,
      leftIdx,
      &projectionImages[leftIdx],
      &projectionImages[rightIdx],
      novelViewGenerators[leftIdx],
      mode
    ));
  }
  for (std::thread& t : threads) { t.join(); }
  // time_checkpoint("first");

  opticalFlowRuntime = getCurrTimeSec() - startOpticalFlowTime;

  // lightfield/parallax formulas
  const float v =
    atanf(FLAGS_zero_parallax_dist / (FLAGS_interpupilary_dist / 2.0f));
  const float psi =
    asinf(sinf(v) * (FLAGS_interpupilary_dist / 2.0f) / cameraRingRadius);
  const float vergeAtInfinitySlabDisplacement =
    psi * (float(camImageWidth) / fovHorizontalRadians);
  const float theta = -M_PI / 2.0f + v + psi;
  const float zeroParallaxNovelViewShiftPixels =
    float(FLAGS_eqr_width) * (theta / (2.0f * M_PI));
  // time_checkpoint("second");
  // ~4.9/23 seconds

  double startNovelViewTime = getCurrTimeSec();
  // a "chunk" will be just the part of the panorama formed from one pair of
  // adjacent cameras. we will stack them horizontally to build the full
  // panorama. we do this so it can be parallelized.
  vector<Mat> panoChunksL(projectionImages.size(), Mat());
  vector<Mat> panoChunksR(projectionImages.size(), Mat());
  vector<std::thread> panoThreads;
  for (int leftIdx = 0; leftIdx < projectionImages.size(); ++leftIdx) {
    panoThreads.push_back(std::thread(
      renderStereoPanoramaChunksThread,
      leftIdx,
      numCams,
      camImageWidth,
      camImageHeight,
      numNovelViews,
      fovHorizontalRadians,
      vergeAtInfinitySlabDisplacement,
      novelViewGenerators[leftIdx],
      &panoChunksL[leftIdx],
      &panoChunksR[leftIdx]
    ));
  }
  for (std::thread& t : panoThreads) { t.join(); }
  // time_checkpoint("third");

  novelViewRuntime = getCurrTimeSec() - startNovelViewTime;

  // time_checkpoint("fourth");
  for (int leftIdx = 0; leftIdx < projectionImages.size(); ++leftIdx) {
    delete novelViewGenerators[leftIdx];
  }
  // time_checkpoint("fifth");

  panoImageL = stackHorizontal(panoChunksL);
  panoImageR = stackHorizontal(panoChunksR);

  panoImageL = offsetHorizontalWrap(panoImageL, zeroParallaxNovelViewShiftPixels);
  panoImageR = offsetHorizontalWrap(panoImageR, -zeroParallaxNovelViewShiftPixels);
  // time_checkpoint("sixth");
  
}

// handles flow between the fisheye top or bottom with the left/right eye side panoramas
void poleToSideFlowThread(
    string eyeName,
    const RigDescription& rig,
    Mat* sideSphericalForEye,
    Mat* fisheyeSpherical,
    Mat* warpedSphericalForEye) {

  // crop the side panorama to the height of the pole image
  Mat croppedSideSpherical = (*sideSphericalForEye)(Rect(0, 0, fisheyeSpherical->cols, fisheyeSpherical->rows));
  croppedSideSpherical = featherAlphaChannel(croppedSideSpherical, FLAGS_std_alpha_feather_size);

  // extend the panoramas and wrap horizontally so we can avoid a seam
  const float kExtendFrac = 1.2f;
  const int extendedWidth = float(fisheyeSpherical->cols) * kExtendFrac;
  Mat extendedSideSpherical(Size(extendedWidth, fisheyeSpherical->rows), CV_8UC4);
  Mat extendedFisheyeSpherical(extendedSideSpherical.size(),  CV_8UC4);
  for (int y = 0; y < extendedSideSpherical.rows; ++y) {
    for (int x = 0; x < extendedSideSpherical.cols; ++x) {
      extendedSideSpherical.at<Vec4b>(y, x) =
        croppedSideSpherical.at<Vec4b>(y, x % fisheyeSpherical->cols);
      extendedFisheyeSpherical.at<Vec4b>(y, x) =
        fisheyeSpherical->at<Vec4b>(y, x % fisheyeSpherical->cols);
    }
  }

  imwriteExceptionOnFail(FLAGS_output_data_dir + "/flow_images/extendedSideSpherical_" + eyeName + ".png", extendedSideSpherical);
  imwriteExceptionOnFail(FLAGS_output_data_dir + "/flow_images/extendedFisheyeSpherical_" + eyeName + ".png", extendedFisheyeSpherical);

  Mat prevFisheyeFlow, prevExtendedSideSpherical, prevExtendedFisheyeSpherical;
  if (FLAGS_prev_frame_data_dir != "NONE") {
    VLOG(1) << "Reading previous frame fisheye flow results from: "
      << FLAGS_prev_frame_data_dir;
    prevFisheyeFlow = readFlowFromFile(
      FLAGS_prev_frame_data_dir + "/flow/flow_" + eyeName + ".bin");
    prevExtendedSideSpherical = imreadExceptionOnFail(
      FLAGS_prev_frame_data_dir + "/flow_images/extendedSideSpherical_" + eyeName + ".png", -1);
    prevExtendedFisheyeSpherical = imreadExceptionOnFail(
      FLAGS_prev_frame_data_dir + "/flow_images/extendedFisheyeSpherical_" + eyeName + ".png", -1);
  }

  Mat flow;
  OpticalFlowInterface* flowAlg = makeOpticalFlowByName(FLAGS_polar_flow_alg);
  flowAlg->computeOpticalFlow(
    extendedSideSpherical,
    extendedFisheyeSpherical,
    prevFisheyeFlow,
    prevExtendedSideSpherical,
    prevExtendedFisheyeSpherical,
    flow,
    OpticalFlowInterface::DirectionHint::DOWN,
    CPU_MODE);
  delete flowAlg;

  VLOG(1) << "Serializing fisheye flow result";
  saveFlowToFile(
    flow,
    FLAGS_output_data_dir + "/flow/flow_" + eyeName + ".bin");

  // make a ramp for alpha/flow magnitude
  const float kRampFrac = 1.0f; // fraction of available overlap used for ramp
  float poleCameraCropRadius;
  float poleCameraRadius;
  float sideCameraRadius;
  if (rig.isNewFormat()) {
    // use fov from bottom camera
    poleCameraRadius = rig.findCameraByDirection(-kGlobalUp).getFov();
    // use fov from first side camera
    sideCameraRadius = approximateFov(rig.rigSideOnly, true);
    // crop is average of side and pole cameras
    poleCameraCropRadius =
      0.5f * (M_PI / 2 - sideCameraRadius) +
      0.5f * (std::min(float(M_PI / 2), poleCameraRadius));
    // convert from radians to degrees
    poleCameraCropRadius *= 180 / M_PI;
    poleCameraRadius *= 180 / M_PI;
    sideCameraRadius *= 180 / M_PI;
  } else {
    CameraMetadata bottom = getBottomCamModel(rig.camModelArrayWithTop);
    const CameraMetadata& side = rig.camModelArray[0];
    poleCameraCropRadius = bottom.fisheyeFovDegreesCrop / 2.0f;
    poleCameraRadius = bottom.fisheyeFovDegrees / 2.0f;
    sideCameraRadius = (side.fovHorizontal / side.aspectRatioWH) / 2.0f;
  }

  const float phiFromPole = poleCameraCropRadius;
  const float phiFromSide = 90.0f - sideCameraRadius;
  const float phiMid = (phiFromPole + phiFromSide) / 2.0f;
  const float phiDiff = fabsf(phiFromPole - phiFromSide);
  const float phiRampStart = phiMid - kRampFrac * phiDiff / 2.0f;
  const float phiRampEnd = phiMid + kRampFrac * phiDiff / 2.0f;

  // ramp for flow magnitude
  //    1               for phi from 0 to phiRampStart
  //    linear drop-off for phi from phiRampStart to phiMid
  //    0               for phi from phiMid to totalRadius
  Mat warp(extendedFisheyeSpherical.size(), CV_32FC2);
  for (int y = 0; y < warp.rows; ++y) {
    const float phi = poleCameraRadius * float(y + 0.5f) / float(warp.rows);
    const float alpha = 1.0f - rampf(phi, phiRampStart, phiMid);
    for (int x = 0; x < warp.cols; ++x) {
      warp.at<Point2f>(y, x) = Point2f(x, y) + (1.0f - alpha) * flow.at<Point2f>(y, x);
    }
  }

  Mat warpedExtendedFisheyeSpherical;
  remap(
    extendedFisheyeSpherical,
    warpedExtendedFisheyeSpherical,
    warp,
    Mat(),
    CV_INTER_CUBIC,
    BORDER_CONSTANT);

  // take the extra strip on the right side and alpha-blend it out on the left side of the result
  *warpedSphericalForEye = warpedExtendedFisheyeSpherical(Rect(0, 0, fisheyeSpherical->cols, fisheyeSpherical->rows));
  int maxBlendX = float(fisheyeSpherical->cols) * (kExtendFrac - 1.0f);
  for (int y = 0; y < warpedSphericalForEye->rows; ++y) {
    for (int x = 0; x < maxBlendX; ++x) {
      const float srcB = warpedSphericalForEye->at<Vec4b>(y, x)[0];
      const float srcG = warpedSphericalForEye->at<Vec4b>(y, x)[1];
      const float srcR = warpedSphericalForEye->at<Vec4b>(y, x)[2];
      const float srcA = warpedSphericalForEye->at<Vec4b>(y, x)[3];
      const float wrapB = warpedExtendedFisheyeSpherical.at<Vec4b>(y, x + fisheyeSpherical->cols)[0];
      const float wrapG = warpedExtendedFisheyeSpherical.at<Vec4b>(y, x + fisheyeSpherical->cols)[1];
      const float wrapR = warpedExtendedFisheyeSpherical.at<Vec4b>(y, x + fisheyeSpherical->cols)[2];
      float alpha = 1.0f - rampf(x, float(maxBlendX) * 0.333f, float(maxBlendX) * 0.667f);
      warpedSphericalForEye->at<Vec4b>(y, x) = Vec4b(
        wrapB * alpha + srcB * (1.0f - alpha),
        wrapG * alpha + srcG * (1.0f - alpha),
        wrapR * alpha + srcR * (1.0f - alpha),
        srcA);
    }
  }

  // make a ramp in the alpha channel for blending with the sides
  //    1               for phi from 0 to phiMid
  //    linear drop-off for phi from phiMid to phiRampEnd
  //    0               for phi from phiRampEnd to totalRadius
  for (int y = 0; y < warp.rows; ++y) {
    const float phi = poleCameraRadius * float(y + 0.5f) / float(warp.rows);
    const float alpha = 1.0f - rampf(phi, phiMid, phiRampEnd);
    for (int x = 0; x < warp.cols; ++x) {
      (*warpedSphericalForEye).at<Vec4b>(y, x)[3] *= alpha;
    }
  }

  copyMakeBorder(
    *warpedSphericalForEye,
    *warpedSphericalForEye,
    0,
    sideSphericalForEye->rows - warpedSphericalForEye->rows,
    0,
    0,
    BORDER_CONSTANT,
    Scalar(0,0,0,0));

  if (FLAGS_save_debug_images) {
    imwriteExceptionOnFail(
      FLAGS_output_data_dir + "/croppedSideSpherical_" + eyeName + ".png",
      croppedSideSpherical);
    imwriteExceptionOnFail(
      FLAGS_output_data_dir + "/warpedSpherical_" + eyeName + ".png",
      *warpedSphericalForEye);
    imwriteExceptionOnFail(
      FLAGS_output_data_dir + "/extendedSideSpherical_" + eyeName + ".png",
      extendedSideSpherical);
  }
}

// does pole removal from the two bottom cameras, and projects the result to equirect
void prepareBottomImagesThread(
    const RigDescription& rig,
    Mat* bottomSpherical) {

  Mat bottomImage;
  if (FLAGS_enable_pole_removal) {
    LOG(INFO) << "Using pole removal masks";
    requireArg(FLAGS_bottom_pole_masks_dir, "bottom_pole_masks_dir");

    float bottomCamUsablePixelsRadius;
    float bottomCam2UsablePixelsRadius;
    bool flip180;
    if (rig.isNewFormat()) {
      const Camera& cam = rig.findCameraByDirection(-kGlobalUp);
      const Camera& cam2 = rig.findLargestDistCamAxisToRigCenter();
      bottomCamUsablePixelsRadius = approximateUsablePixelsRadius(cam);
      bottomCam2UsablePixelsRadius = approximateUsablePixelsRadius(cam2);
      flip180 = cam.up().dot(cam2.up()) < 0 ? true : false;
    } else {
      const CameraMetadata& cam = getBottomCamModel(rig.camModelArrayWithTop);
      const CameraMetadata& cam2 = getBottomCamModel2(rig.camModelArrayWithTop);
      bottomCamUsablePixelsRadius = cam.usablePixelsRadius;
      bottomCam2UsablePixelsRadius = cam2.usablePixelsRadius;
      flip180 = cam2.flip180;
    }
    combineBottomImagesWithPoleRemoval(
      FLAGS_imgs_dir,
      FLAGS_bottom_pole_masks_dir,
      FLAGS_prev_frame_data_dir,
      FLAGS_output_data_dir,
      FLAGS_save_debug_images,
      true, // save data that will be used in the next frame
      FLAGS_poleremoval_flow_alg,
      FLAGS_std_alpha_feather_size,
      FLAGS_enable_render_coloradjust,
      rig.getBottomCameraId(),
      rig.getBottomCamera2Id(),
      bottomCamUsablePixelsRadius,
      bottomCam2UsablePixelsRadius,
      flip180,
      bottomImage);
  } else {
    LOG(INFO) << "Using primary bottom camera";
    const string bottomImageFilename = rig.getBottomCameraId() + ".png";
    const string bottomImagePath = FLAGS_imgs_dir + "/" + bottomImageFilename;
    bottomImage = imreadExceptionOnFail(bottomImagePath, CV_LOAD_IMAGE_COLOR);
  }

  if (rig.isNewFormat()) {
    const Camera& camera = rig.findCameraByDirection(-kGlobalUp);
    bottomSpherical->create(
      FLAGS_eqr_height * camera.getFov() / M_PI,
      FLAGS_eqr_width,
      CV_8UC3);
    bicubicRemapToSpherical(
      *bottomSpherical,
      bottomImage,
      camera,
      0,
      2.0f * M_PI,
      -(M_PI / 2.0f),
      -(M_PI / 2.0f - camera.getFov()));
  } else {
    CameraMetadata bottom = getBottomCamModel(rig.camModelArrayWithTop);
    *bottomSpherical = bicubicRemapFisheyeToSpherical(
      bottom,
      bottomImage,
      Size(
        FLAGS_eqr_width,
        FLAGS_eqr_height * (bottom.fisheyeFovDegrees / 2.0f) / 180.0f));
  }

  // if we skipped pole removal, there is no alpha channel and we need to add one.
  if (bottomSpherical->type() != CV_8UC4) {
    cvtColor(*bottomSpherical, *bottomSpherical, CV_BGR2BGRA);
  }

  // the alpha channel in bottomSpherical is the result of pole removal/flow. this can in
  // some cases cause an alpha-channel discontinuity at the boundary of the image, which
  // will have an effect on flow between bottom and sides. to mitigate that, we do another
  // pass of feathering on bottomSpherical before converting to polar coordinates.
  const int yFeatherStart = bottomSpherical->rows - 1 - FLAGS_std_alpha_feather_size;
  for (int y = yFeatherStart; y < bottomSpherical->rows; ++y) {
    for (int x = 0; x < bottomSpherical->cols; ++x) {
      const float alpha =
        1.0f - float(y - yFeatherStart) / float(FLAGS_std_alpha_feather_size);
      bottomSpherical->at<Vec4b>(y, x)[3] =
        min(bottomSpherical->at<Vec4b>(y, x)[3], (unsigned char)(255.0f * alpha));
    }
  }

  if (FLAGS_save_debug_images) {
    imwriteExceptionOnFail(FLAGS_output_data_dir + "/_bottomSpherical.png", *bottomSpherical);
  }
}

// similar to prepareBottomImagesThread but there is no pole removal
void prepareTopImagesThread(
    const RigDescription& rig,
    Mat* topSpherical) {

  const string topImageFilename = rig.getTopCameraId() + ".png";
  const string topImagePath = FLAGS_imgs_dir + "/" + topImageFilename;
  Mat topImage = imreadExceptionOnFail(topImagePath, CV_LOAD_IMAGE_COLOR);
  if (rig.isNewFormat()) {
    const Camera& camera = rig.findCameraByDirection(kGlobalUp);
    topSpherical->create(
      FLAGS_eqr_height * camera.getFov() / M_PI,
      FLAGS_eqr_width,
      CV_8UC3);
    bicubicRemapToSpherical(
      *topSpherical,
      topImage,
      camera,
      2.0f * M_PI,
      0,
      M_PI / 2.0f,
      M_PI / 2.0f - camera.getFov());
  } else {
    CameraMetadata top = getTopCamModel(rig.camModelArrayWithTop);
    *topSpherical = bicubicRemapFisheyeToSpherical(
      top,
      topImage,
      Size(
        FLAGS_eqr_width,
        FLAGS_eqr_height * (top.fisheyeFovDegrees / 2.0f) / 180.0f));

  }

  // alpha feather the top spherical image for flow purposes
  cvtColor(*topSpherical, *topSpherical, CV_BGR2BGRA);
  const int yFeatherStart = topSpherical->rows - 1 - FLAGS_std_alpha_feather_size;
  for (int y = yFeatherStart ; y < topSpherical->rows ; ++y) {
    for (int x = 0; x < topSpherical->cols; ++x) {
      const float alpha =
        1.0f - float(y - yFeatherStart) / float(FLAGS_std_alpha_feather_size);
      topSpherical->at<Vec4b>(y, x)[3] = 255.0f * alpha;
    }
  }

  if (FLAGS_save_debug_images) {
    imwriteExceptionOnFail(FLAGS_output_data_dir + "/_topSpherical.png", *topSpherical);
  }
}

// sharpen the left or right eye panorama using a periodic boundary
void sharpenThread(Mat* sphericalImage) {
 
   Mat lowPassSphericalImage(sphericalImage->rows, sphericalImage->cols, CV_8UC3);

#ifdef GPU_SHARPEN
   //*sphericalImage = lowPassSphericalImage;
  double startTimeGPU = surround360::util::getCurrTimeSec();
   IIRFilter(
   *sphericalImage, 0.25f, lowPassSphericalImage);
   Mat sMat = lowPassSphericalImage;
  double endTimeGPU = surround360::util::getCurrTimeSec();
//  if(!sMat.empty()){
//  imshow("Image",sMat);
//  }
//  else
//  {
//  LOG(INFO) << "Image is Empty" ;
//  }
//  cvWaitKey(0);
#endif
// 
#ifdef CPU_SHARPEN
  const WrapBoundary<float> wrapB;
  const ReflectBoundary<float> reflectB;
  double startTimeCPU = surround360::util::getCurrTimeSec();
  //Mat lowPassSphericalImage(sphericalImage->rows, sphericalImage->cols, CV_8UC3);
  
  iirLowPass<WrapBoundary<float>, ReflectBoundary<float>, Vec3b>(
    *sphericalImage, 0.25f, lowPassSphericalImage, wrapB, reflectB);
  
  sharpenWithIirLowPass<Vec3b>(
    *sphericalImage, lowPassSphericalImage, 1.0f + FLAGS_sharpenning);
  
  double endTimeCPU = surround360::util::getCurrTimeSec() - startTimeCPU; 
  LOG(INFO) << "Time in Sharpening CPU is " <<  endTimeCPU ;
#endif
//  
#ifdef GPU_SHARPEN
//  double startTimeGPU = surround360::util::getCurrTimeSec();
//  cvtColor(sMat, sMat, CV_BGR2BGRA);
//  Mat result;
//  result = sharpenGPU(sMat);
  //Mat color;
  //cvtColor(sMat, color , CV_BGRA2BGR, 3);
//#ifndef CPU_SHARPEN
//  *sphericalImage = color;
//#endif
  LOG(INFO) << "Time in Sharpening GPU is " <<  endTimeGPU - startTimeGPU;
#endif 
//
// // Correctness for Sharpening Below.
// // Can add SSIM here 
// // Compare for SSIM between '*sphericalImage' and 'color' .
#if defined(CPU_SHARPEN) && defined(GPU_SHARPEN)  
  Mat Testspherical;
  //cv::Scalar ssim_sharp = getMSSIM(*sphericalImage, sMat);
  Mat diff;
  cvtColor(*sphericalImage, Testspherical, CV_BGR2GRAY);
  cvtColor( sMat, sMat, CV_BGR2GRAY);
  bitwise_xor(sMat, Testspherical, diff);
  bool eq = (cv::countNonZero(diff) == 0 );

  //LOG(INFO) << "SHARPENING correctness: ssim_sharpening " << ssim_sharp ;
  if (eq) 
     LOG(INFO) << "Sharpening Correctness Passed"; 
  else
    {
     float resolution = 7680.0*4320.0;  // Applies only for 8K resolution, need to hardcode for 6K resolution if required.
     LOG(INFO) << "Sharpening Correctness Failed "; 
     LOG(INFO) << "Percentage Difference is " << cv::countNonZero(diff)*100/resolution; 
    }
#endif  


}

// If the un-padded height is odd and targetHeight is even, we can't do equal
// padding to get the final image to be targetHeight. the formulas below give
// equal padding if possible, or equal +/-1 if not.
void padToheight(Mat& unpaddedImage, const int targetHeight) {
  const int paddingAbove = (targetHeight - unpaddedImage.rows) / 2;
  const int paddingBelow = targetHeight - unpaddedImage.rows - paddingAbove;
  copyMakeBorder(
    unpaddedImage,
    unpaddedImage,
    paddingAbove,
    paddingBelow,
    0,
    0,
    BORDER_CONSTANT,
    Scalar(0.0, 0.0, 0.0));
}

// run the whole stereo panorama rendering pipeline
void renderStereoPanorama() {
  requireArg(FLAGS_rig_json_file, "rig_json_file");
  requireArg(FLAGS_imgs_dir, "imgs_dir");
  requireArg(FLAGS_output_data_dir, "output_data_dir");
  requireArg(FLAGS_output_equirect_path, "output_equirect_path");

  const double startTime = getCurrTimeSec();

  ColorAdjustmentSampleLogger::instance().enabled =
    FLAGS_enable_render_coloradjust;

  RigDescription rig(FLAGS_rig_json_file, FLAGS_new_rig_format);
  
  // prepare the bottom camera(s) by doing pole removal and projections in a thread.
  // will join that thread as late as possible.
  Mat bottomSpherical;
  std::thread prepareBottomThread;
  if (FLAGS_enable_bottom) {
    VLOG(1) << "Bottom cameras enabled. Preparing bottom projections in a thread";
    prepareBottomThread = std::thread(
      prepareBottomImagesThread,
      std::cref(rig),
      &bottomSpherical);
  }

  // top cameras are handled similar to bottom cameras- do anything we can in a thread
  // that is joined as late as possible.
  Mat topSpherical;
  std::thread prepareTopThread;
  if (FLAGS_enable_top) {
    prepareTopThread = std::thread(
      prepareTopImagesThread,
      cref(rig),
      &topSpherical);
  }

  // projection to spherical coordinates
  vector<Mat> projectionImages;

  if (FLAGS_save_debug_images) {
    system(string("rm -f " + FLAGS_output_data_dir + "/projections/*").c_str());
  }

  const double startProjectSphericalTime = getCurrTimeSec();
  LOG(INFO) << "Projecting camera images to spherical";
  projectSphericalCamImages(rig, FLAGS_imgs_dir, projectionImages);
  const double endProjectSphericalTime = getCurrTimeSec();

  // generate novel views and stereo spherical panoramas
  double opticalFlowRuntime, novelViewRuntime;
  Mat sphericalImageL, sphericalImageR;
  LOG(INFO) << "Rendering stereo panorama";
  // time_checkpoint("");
  // ~20/36 seconds hotspot

  const double fovHorizontal = rig.isNewFormat()
    ? 2 * approximateFov(rig.rigSideOnly, false) * (180 / M_PI)
    : rig.camModelArray[0].fovHorizontal;
  generateRingOfNovelViewsAndRenderStereoSpherical(
    rig.getRingRadius(),
    fovHorizontal,
    projectionImages,
    sphericalImageL,
    sphericalImageR,
    opticalFlowRuntime,
    novelViewRuntime);
  // time_checkpoint("first");
  LOG(INFO) << "Finished Multithreaded Run" ;
  LOG(INFO) << "MT time: Optical Flow: " << opticalFlowRuntime ;
  LOG(INFO) << "MT time: Novel View: " << novelViewRuntime ;

  double st_opticalFlowRuntime, st_novelViewRuntime;
  Mat st_sphericalImageL, st_sphericalImageR;
  st_generateRingOfNovelViewsAndRenderStereoSpherical(
    rig.getRingRadius(),
    fovHorizontal,
    projectionImages,
    st_sphericalImageL,
    st_sphericalImageR,
    st_opticalFlowRuntime,
    st_novelViewRuntime);

  // Checking correctness
  Mat st_sphericalImageL_cn1, st_sphericalImageR_cn1;
  Mat sphericalImageL_cn1, sphericalImageR_cn1;
  Mat diff1, diff2;
  cvtColor(sphericalImageL, sphericalImageL_cn1, CV_RGB2GRAY);
  cvtColor(sphericalImageR, sphericalImageR_cn1, CV_RGB2GRAY);
  cvtColor(st_sphericalImageL, st_sphericalImageL_cn1, CV_RGB2GRAY);
  cvtColor(st_sphericalImageR, st_sphericalImageR_cn1, CV_RGB2GRAY);
  cv::bitwise_xor(sphericalImageL_cn1, st_sphericalImageL_cn1, diff1);
  cv::bitwise_xor(sphericalImageR_cn1, st_sphericalImageR_cn1, diff2);
  bool eq1 = (cv::countNonZero(diff1) == 0);
  bool eq2 = (cv::countNonZero(diff2) == 0);

  cv::Scalar ssimL = getMSSIM(sphericalImageL, st_sphericalImageL);
  cv::Scalar ssimR = getMSSIM(sphericalImageR, st_sphericalImageR);
  LOG(INFO) << "OPT correctness: ssim: L" << ssimL << " R: " << ssimR ;

  if (eq1 && eq2) {
    LOG(INFO) << "Finished SingleThreaded Run" ;
    LOG(INFO) << "Correctness passed" ; 
    LOG(INFO) << "ST time: Optical Flow: " << st_opticalFlowRuntime ;
    LOG(INFO) << "ST time: Novel View: " << st_novelViewRuntime ;
  } 
  else {
    float resolution = 7680.0*4320.0;  // Applies only for 8K resolution, need to hardcode for 6K resolution if required.
    LOG(INFO) << "Correctness Failed in MT vs ST" ; 
    LOG(INFO) << "Image1L and Image2L differ in " << cv::countNonZero(diff1)*100.0/resolution << " pixels " ;
    LOG(INFO) << "Image1R and Image2R differ in " << cv::countNonZero(diff2)*100.0/resolution << " pixels " ;
    LOG(INFO) << "ST time: Optical Flow: " << st_opticalFlowRuntime ;
    LOG(INFO) << "ST time: Novel View: " << st_novelViewRuntime ;
  }

  if (FLAGS_save_debug_images) {
    VLOG(1) << "Offset-warping images for debugging";
    Mat wrapSphericalImageL, wrapSphericalImageR;
    wrapSphericalImageL = offsetHorizontalWrap(sphericalImageL, sphericalImageL.cols/3);
    wrapSphericalImageR = offsetHorizontalWrap(sphericalImageR, sphericalImageR.cols/3);
    imwriteExceptionOnFail(FLAGS_output_data_dir + "/sphericalImgL.png", sphericalImageL);
    imwriteExceptionOnFail(FLAGS_output_data_dir + "/sphericalImgR.png", sphericalImageR);
    imwriteExceptionOnFail(FLAGS_output_data_dir + "/sphericalImg_offsetwrapL.png", wrapSphericalImageL);
    imwriteExceptionOnFail(FLAGS_output_data_dir + "/sphericalImg_offsetwrapR.png", wrapSphericalImageR);
  }

  // so far we only operated on the strip that contains the full vertical FOV of
  // the side cameras. before merging those results with top/bottom cameras,
  // we will pad the side images out to be a full 180 degree vertical equirect.
  padToheight(sphericalImageL, FLAGS_eqr_height);
  padToheight(sphericalImageR, FLAGS_eqr_height);

  // if both top and bottom cameras are enabled, there are 4 threads that can be done in
  // parallel (for top/bottom, we flow to the left eye and right eye side panoramas).
  std::thread topFlowThreadL, topFlowThreadR, bottomFlowThreadL, bottomFlowThreadR;
  const double topBottomToSideStartTime = getCurrTimeSec();

  // if we have a top camera, do optical flow with its image and the side camera
  Mat topSphericalWarpedL, topSphericalWarpedR;
  if (FLAGS_enable_top) {
    prepareTopThread.join(); // this is the latest we can wait

    topFlowThreadL = std::thread(
      poleToSideFlowThread,
      "top_left",
      cref(rig),
      &sphericalImageL,
      &topSpherical,
      &topSphericalWarpedL);

    topFlowThreadR = std::thread(
      poleToSideFlowThread,
      "top_right",
      cref(rig),
      &sphericalImageR,
      &topSpherical,
      &topSphericalWarpedR);
  }

  Mat flipSphericalImageL, flipSphericalImageR;
  Mat bottomSphericalWarpedL, bottomSphericalWarpedR;
  if (FLAGS_enable_bottom) {
    prepareBottomThread.join(); // this is the latest we can wait

    // flip the side images upside down for bottom flow
    flip(sphericalImageL, flipSphericalImageL, -1);
    flip(sphericalImageR, flipSphericalImageR, -1);

    bottomFlowThreadL = std::thread(
      poleToSideFlowThread,
      "bottom_left",
      cref(rig),
      &flipSphericalImageL,
      &bottomSpherical,
      &bottomSphericalWarpedL);

    bottomFlowThreadR = std::thread(
      poleToSideFlowThread,
      "bottom_right",
      cref(rig),
      &flipSphericalImageR,
      &bottomSpherical,
      &bottomSphericalWarpedR);
  }

  // now that all 4 possible threads have been spawned, we are ready to wait for the
  // threads to finish, then composite the results
  if (FLAGS_enable_top) {
    topFlowThreadL.join();
    topFlowThreadR.join();

    if (FLAGS_enable_render_coloradjust) {
      sphericalImageL = flattenLayersDeghostPreferBaseAdjustBrightness(
        sphericalImageL, topSphericalWarpedL);
      sphericalImageR = flattenLayersDeghostPreferBaseAdjustBrightness(
        sphericalImageR, topSphericalWarpedR);
    } else {
      sphericalImageL = flattenLayersDeghostPreferBase(
        sphericalImageL, topSphericalWarpedL);
      sphericalImageR = flattenLayersDeghostPreferBase(
        sphericalImageR, topSphericalWarpedR);
    }
  }

  if (FLAGS_enable_bottom) {
    bottomFlowThreadL.join();
    bottomFlowThreadR.join();

    flip(sphericalImageL, sphericalImageL, -1);
    flip(sphericalImageR, sphericalImageR, -1);
    if (FLAGS_enable_render_coloradjust) {
      sphericalImageL = flattenLayersDeghostPreferBaseAdjustBrightness(
        sphericalImageL, bottomSphericalWarpedL);
      sphericalImageR = flattenLayersDeghostPreferBaseAdjustBrightness(
        sphericalImageR, bottomSphericalWarpedR);
    } else {
      sphericalImageL = flattenLayersDeghostPreferBase(
        sphericalImageL, bottomSphericalWarpedL);
      sphericalImageR = flattenLayersDeghostPreferBase(
        sphericalImageR, bottomSphericalWarpedR);
    }
    flip(sphericalImageL, sphericalImageL, -1);
    flip(sphericalImageR, sphericalImageR, -1);
  }
  const double topBottomToSideEndTime = getCurrTimeSec();

  // depending on how things are handled, we might still have an alpha channel.
  // if so, flatten the image to 3 channel
  if (sphericalImageL.type() != CV_8UC3) {
    VLOG(1) << "Flattening from 4 channels to 3 channels";
    cvtColor(sphericalImageL, sphericalImageL, CV_BGRA2BGR);
    cvtColor(sphericalImageR, sphericalImageR, CV_BGRA2BGR);
  }

  if (FLAGS_save_debug_images) {
    imwriteExceptionOnFail(FLAGS_output_data_dir + "/eqr_sideL.png", sphericalImageL);
    imwriteExceptionOnFail(FLAGS_output_data_dir + "/eqr_sideR.png", sphericalImageR);
  }
  // time_checkpoint("second");
  // ~4.7/36 seconds hotspot

  const double startSharpenTime = getCurrTimeSec();
  if (FLAGS_sharpenning > 0.0f) {
    VLOG(1) << "Sharpening";
    std::thread sharpenThreadL(sharpenThread, &sphericalImageL);
    std::thread sharpenThreadR(sharpenThread, &sphericalImageR);
    sharpenThreadL.join();
    sharpenThreadR.join();
    if (FLAGS_save_debug_images) {
      imwriteExceptionOnFail(FLAGS_output_data_dir + "/_eqr_sideL_sharpened.png", sphericalImageL);
      imwriteExceptionOnFail(FLAGS_output_data_dir + "/_eqr_sideR_sharpened.png", sphericalImageR);
    }
  }
  const double endSharpenTime = getCurrTimeSec();
  // time_checkpoint("third");

  // project the horizontal panoramas to cubemaps and composite the top
  const double startCubemapTime = getCurrTimeSec();
  if (FLAGS_cubemap_width > 0 && FLAGS_cubemap_height > 0
      && !FLAGS_output_cubemap_path.empty()) {
    LOG(INFO) << "Generating stereo cubemap";
    Mat cubemapImageL = stackOutputCubemapFaces(
        FLAGS_cubemap_format,
        convertSphericalToCubemapBicubicRemap(
          sphericalImageL,
          M_PI,
          FLAGS_cubemap_width,
          FLAGS_cubemap_height));
    Mat cubemapImageR = stackOutputCubemapFaces(
        FLAGS_cubemap_format, convertSphericalToCubemapBicubicRemap(
          sphericalImageR,
          M_PI,
          FLAGS_cubemap_width,
          FLAGS_cubemap_height));
    Mat stereoCubemap = stackVertical(vector<Mat>({cubemapImageL, cubemapImageR}));
    imwriteExceptionOnFail(FLAGS_output_cubemap_path, stereoCubemap);
  }
  const double endCubemapTime = getCurrTimeSec();
  // time_checkpoint("fourth");

  if (FLAGS_final_eqr_width != 0 &&
      FLAGS_final_eqr_height != 0 &&
      FLAGS_final_eqr_width != FLAGS_eqr_width &&
      FLAGS_final_eqr_height != FLAGS_eqr_height / 2) {
    VLOG(1) << "Resizing before final equirect stack (for proper video size)";
    resize(
      sphericalImageL,
      sphericalImageL,
      Size(FLAGS_final_eqr_width, FLAGS_final_eqr_height / 2),
      0,
      0,
      INTER_CUBIC);
    resize(
      sphericalImageR,
      sphericalImageR,
      Size(FLAGS_final_eqr_width, FLAGS_final_eqr_height / 2),
      0,
      0,
      INTER_CUBIC);
  }
  // time_checkpoint("fifth");

  LOG(INFO) << "Creating stereo equirectangular image";
  Mat stereoEquirect = stackVertical(vector<Mat>({sphericalImageL, sphericalImageR}));
  imwriteExceptionOnFail(FLAGS_output_equirect_path, stereoEquirect);

  if (FLAGS_enable_render_coloradjust &&
      !FLAGS_brightness_adjustment_dest.empty()) {
    LOG(INFO) << "running side brightness adjustment";
    ColorAdjustmentSampleLogger& colorSampleLogger =
      ColorAdjustmentSampleLogger::instance();

    LOG(INFO) << "# color samples = " << colorSampleLogger.samples.size();

    vector<double> sideCamBrightnessAdjustments =
      computeBrightnessAdjustmentsForSideCameras(
        rig.getSideCameraCount(), colorSampleLogger.samples);

    LOG(INFO) << "writing brightness adjustments to file: "
      << FLAGS_brightness_adjustment_dest;
    ofstream brightnessAdjustFile(FLAGS_brightness_adjustment_dest);
    if (!brightnessAdjustFile) {
      throw VrCamException(
        "file write failed: " + FLAGS_brightness_adjustment_dest);
    }
    vector<string> brightnessStrs;
    for (int i = 0; i < sideCamBrightnessAdjustments.size(); ++i) {
      brightnessStrs.push_back(to_string(sideCamBrightnessAdjustments[i]));
    }
    brightnessAdjustFile << stringJoin(",", brightnessStrs);
    brightnessAdjustFile.close();
  }

  const double endTime = getCurrTimeSec();
  VLOG(1) << "--- Runtime breakdown (sec) ---";
  VLOG(1) << "Total:\t\t\t" << endTime - startTime;
  VLOG(1) << "Spherical projection:\t" << endProjectSphericalTime - startProjectSphericalTime;
  VLOG(1) << "Side optical flow:\t\t" << opticalFlowRuntime;
  VLOG(1) << "Novel view panorama:\t" << novelViewRuntime;
  VLOG(1) << "Flow top+bottom with sides:\t" << topBottomToSideEndTime - topBottomToSideStartTime;
  VLOG(1) << "Sharpen:\t\t" << endSharpenTime - startSharpenTime;
  VLOG(1) << "Equirect -> Cubemap:\t" << endCubemapTime - startCubemapTime;
}

int main(int argc, char** argv) {
  initSurround360(argc, argv);
  renderStereoPanorama();
  return EXIT_SUCCESS;
}
