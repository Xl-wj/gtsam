#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <vector>
#include <glog/logging.h>
#include <fstream>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Symbol.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include "tic_toc.hpp"

using namespace std;
using namespace gtsam;
using namespace external::times;


vector<string> stringSplit(const string& s, const string& delimiter) {
  size_t pos1 = 0;
  size_t pos2 = 0;

  vector<string> result;
  while ((pos2 = s.find(delimiter, pos1)) != string::npos) {
    if (pos1 != pos2) result.emplace_back(s.substr(pos1, pos2 - pos1));

    pos1 = pos2 + delimiter.length();
  }
  if (pos1 != s.size()) result.push_back(s.substr(pos1));
  return result;
}

struct SubmapFactor : std::pair<uint16_t, uint16_t>{
    Eigen::Affine3d Affine() const {
      return Eigen::Affine3d(Eigen::Translation3d(t) * q);
    }
    Eigen::Vector3d t;
    Eigen::Quaterniond q;
    int id;
//    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
struct SubmapFactorKey : public std::unary_function<SubmapFactor, uint32_t> {
    uint32_t operator()(const SubmapFactor& f) const {
      return (uint32_t(f.first << 16) | uint32_t(f.second));
    }
};

typedef std::vector<SubmapFactor, Eigen::aligned_allocator<SubmapFactor>> Factors;

Factors LoadSubmapFactors(const std::string &factor_file) {
  Factors factors(0);

  ifstream pose_file(factor_file);
  if(!pose_file.is_open())
    LOG(ERROR) << "Failed to open the pose file: " << factor_file;

  string temp;
  while(getline(pose_file, temp)) {
    auto strs = stringSplit(temp, " ");

    if (strs.size() != 9)
      LOG(ERROR) << "Find invalide pose line in given file! pose_path: " << factor_file;

    factors.emplace_back();
    factors.back().first = uint16_t(stoi(strs[0]));
    factors.back().second = uint16_t(stoi(strs[1]));
    factors.back().t = Eigen::Vector3d(stod(strs[2]), stod(strs[3]), stod(strs[4]));
    factors.back().q = Eigen::Quaterniond(stod(strs[8]), stod(strs[5]), stod(strs[6]), stod(strs[7]));
  }
  return factors;
}

struct PoseTypeWithId {
    Eigen::Quaterniond q;
    Eigen::Vector3d t;
    double timestamp;
    size_t id;

    Eigen::Affine3d Affine() const {
      return Eigen::Affine3d(Eigen::Translation3d(t) * q);
    }

    void SetOri(const Eigen::Affine3d &ref) {
      auto rel = ref.inverse() * Affine();
      q = rel.linear();
      t = rel.translation();
    }

    void SetIdentity() {
      Eigen::Affine3d rel = Eigen::Affine3d::Identity();
      q = rel.linear();
      t = rel.translation();
    }
};
typedef std::vector<PoseTypeWithId, Eigen::aligned_allocator<PoseTypeWithId>> PosesType;


PosesType LoadInterpolationTimestamp(const std::string &pose_path) {
  PosesType poses;

  ifstream pose_file(pose_path);
  if(!pose_file.is_open())
    LOG(ERROR) << "Failed to open the pose file: " << pose_path;

  string temp;
  while(getline(pose_file, temp)) {
    auto strs = stringSplit(temp, " ");

    if (strs.size() != 12)
      LOG(ERROR) << "Find invalide pose line in given file! pose_path: " << pose_path;

    poses.emplace_back();
    poses.back().id = stod(strs[0]);
    poses.back().timestamp = stod(strs[1]);
    poses.back().q = Eigen::Quaterniond(stod(strs[8]), stod(strs[5]), stod(strs[6]), stod(strs[7]));
    poses.back().t = Eigen::Vector3d(stod(strs[2]), stod(strs[3]), stod(strs[4]));
  }

  return poses;
}

void SavePosesType2File(const PosesType &poses, const std::string &pose_file) {

  std::ofstream fs(pose_file);
  for(const auto &pose : poses) {
    fs << std::fixed << std::setprecision(12)
       << pose.id << " "
       << pose.timestamp << " "
       << pose.t[0] << " "
       << pose.t[1] << " "
       << pose.t[2] << " "
       << pose.q.x() << " "
       << pose.q.y() << " "
       << pose.q.z() << " "
       << pose.q.w() << " "
       << 0.0 << " "
       << 0.0 << " "
       << 0.0 << std::endl;
  }
  LOG(INFO) << "Finished to save trajectory to file: " << pose_file;
}


Eigen::Affine3d GetRelativeTF(const PoseTypeWithId &from, const PoseTypeWithId &to) {
  return from.Affine().inverse() * to.Affine();
}
static Eigen::Vector3d ToYPRAngles(const Eigen::Matrix3d &r) {
  Eigen::Vector3d ypr;

  if (r(2, 0) < 1) {
    if (r(2, 0) > -1) {
      ypr.y() = std::asin(-r(2, 0));
      ypr.x() = std::atan2(r(1, 0), r(0, 0));
      ypr.z() = std::atan2(r(2, 1), r(2, 2));
    } else {
      ypr.y() = M_PI_2;
      ypr.x() = -std::atan2(-r(1, 2), r(1, 1));
      ypr.z() = 0.0;
    }
  } else {
    ypr.y() = -M_PI_2;
    ypr.x() = std::atan2(-r(1, 2), r(1, 1));
    ypr.z() = 0.0;
  }

  return ypr;
}


void CheckFactorsUsingPoses(const Factors &factors, const PosesType &poses) {

  for(const auto &factor : factors) {
    size_t id_1 = factor.first;
    size_t id_2 = factor.second;

    CHECK_EQ(poses[id_1 - 1].id, id_1);
    CHECK_EQ(poses[id_2 - 1].id, id_2);

    auto relative = GetRelativeTF(poses[id_1 - 1], poses[id_2 - 1]);
    Eigen::Affine3d error = relative.inverse() * factor.Affine();

    LOG(INFO) << std::fixed << std::setprecision(10)
              << "error: " << std::endl << error.matrix() << endl
              << "trans: " << std::endl << error.translation() << endl
              << "linear: " << std::endl << 180.0 * ToYPRAngles(error.linear().matrix()) / M_PI;
  }

}

void DumpEstimateResults2File(const Values &values, const std::string &file) {
  // convert
  PosesType poses_out;
  const auto keys = values.keys();
  for (const auto& key : keys) {
    std::string key_str = DefaultKeyFormatter(key);
    Eigen::Affine3d opt_pose(values.at(key).cast<gtsam::Pose3>().matrix());

    poses_out.emplace_back();
    poses_out.back().id = size_t(stoi(key_str.substr(1, key_str.size()-1)));
    poses_out.back().t = opt_pose.translation();
    poses_out.back().q = opt_pose.linear();
  }

  LOG(INFO) << "poses_out.size = " << poses_out.size();
  for(size_t i=1; i<poses_out.size(); i++) {
    poses_out[i].SetOri(poses_out[0].Affine());
  }
  poses_out[0].SetIdentity();

  SavePosesType2File(poses_out, file);
}

gtsam::Pose3 PoseTypeWithId2GtsamType(const PoseTypeWithId &pose) {
  auto ypr = ToYPRAngles(Eigen::Matrix3d(pose.q));
  return Pose3(Rot3::RzRyRx(ypr.z(), ypr.y(), ypr.x()), Point3(pose.t));
}



void Optimizer(const Factors &factors, const PosesType &poses, const PosesType &icp_poses) {
  NonlinearFactorGraph graph;

  gtsam::Vector Vector6(6);
  Vector6 << 1.0, 1.0, 0.9, 1e-1, 1e-1, 1e-1;
  noiseModel::Diagonal::shared_ptr prior_noise = noiseModel::Diagonal::Variances(Vector6);
  Vector6 << 0.05, 0.05, 0.05, 1e-4, 1e-4, 1e-4;
  noiseModel::Diagonal::shared_ptr icp_noise = noiseModel::Diagonal::Sigmas(Vector6);
  noiseModel::mEstimator::Huber::Create(1.0);

  Values initials;

  // add prior from rtk
  for(const auto &icp_pose : icp_poses) {
    auto rtk_pose = poses[icp_pose.id - 1];
    CHECK_EQ(rtk_pose.id, icp_pose.id);

    gtsam::Pose3 pose = PoseTypeWithId2GtsamType(rtk_pose);

    graph.add(PriorFactor<Pose3>(Symbol('x', rtk_pose.id), pose, prior_noise));
    initials.insert(Symbol('x', rtk_pose.id), pose);
  }

  // add icp constrains
  for(size_t i = 0; i < icp_poses.size() - 1; i++) {

    gtsam::Pose3 poseFrom = PoseTypeWithId2GtsamType(icp_poses[i]);
    gtsam::Pose3 poseTo = PoseTypeWithId2GtsamType(icp_poses[i + 1]);

    graph.add(BetweenFactor<Pose3>(Symbol('x', icp_poses[i].id),
                                   Symbol('x', icp_poses[i + 1].id),
                                   poseFrom.between(poseTo), icp_noise));
  }

//  graph.print("\n Factor Graph: \n\n");
//  initials.print("\n initials values: \n");

  ISAM2Params isam2Params;
  isam2Params.relinearizeThreshold = 0.01;
  isam2Params.relinearizeSkip = 1;
  ISAM2 *isam = new ISAM2(isam2Params);

  TicToc t;
  isam->update(graph, initials);
  LOG(INFO) << "iSam Inference factor graph time: " << t.toc();
  Values isam_values = isam->calculateEstimate();
  std::string isam_out_pose_file = "/home/xl/align_map/submap_align/graph_pose_sam.txt";
  DumpEstimateResults2File(isam_values, isam_out_pose_file);


  TicToc t_lm;
  Values lm_result = LevenbergMarquardtOptimizer(graph, initials).optimize();
  LOG(INFO) << "LM Inference factor graph time: " << t_lm.toc();
  std::string lm_out_pose_file = "/home/xl/align_map/submap_align/graph_pose_lm.txt";
  DumpEstimateResults2File(lm_result, lm_out_pose_file);


  TicToc t_gn;
  Values gn_result = GaussNewtonOptimizer(graph, initials).optimize();
  LOG(INFO) << "GN Inference factor graph time: " << t_gn.toc();
  std::string gn_out_pose_file = "/home/xl/align_map/submap_align/graph_pose_gn.txt";
  DumpEstimateResults2File(gn_result, gn_out_pose_file);

}

int main(int argc, char** argv) {
//  const std::string factor_file = "/home/xl/align_map/submap_align/factors.txt";
//  const std::string poses_file  = "/home/xl/jiashan_test/0819_map_1/location/interpolation_out.txt";
//  const std::string icp_poses_file  = "/home/xl/align_map/submap_align/pose.txt";
//
//  Factors factors = LoadSubmapFactors(factor_file);
//  PosesType poses = LoadInterpolationTimestamp(poses_file);
//  PosesType icp_poses = LoadInterpolationTimestamp(icp_poses_file);

//  CheckFactorsUsingPoses(factors, poses);
//  Optimizer(factors, poses, icp_poses);
  return 0;
}
