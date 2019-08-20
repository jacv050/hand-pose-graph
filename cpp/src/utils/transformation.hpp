#ifndef __transformation__
#define __transformation__

#include <Eigen/Geometry>
#include <jsoncpp/json/json.h>

namespace qutils{

Eigen::Quaternionf json2pquat(const Json::Value& p){
  return Eigen::Quaternionf(0, p[0].asFloat(), p[1].asFloat(), p[2].asFloat());
}

Eigen::Quaternionf json2rquat(const Json::Value& p){
  return Eigen::Quaternionf(p[0].asFloat(), p[1].asFloat(), p[2].asFloat(), p[3].asFloat());
}

Eigen::Quaternionf euler2quaternionYXZ(const std::vector<float>& rot){
  return Eigen::AngleAxisf(rot[1], Eigen::Vector3f::UnitY())
             * Eigen::AngleAxisf(rot[2], Eigen::Vector3f::UnitX())
             * Eigen::AngleAxisf(rot[0], Eigen::Vector3f::UnitZ());
}

void world2camera(const std::vector<float>& p_pos,
    const std::vector<float>& p_rot,
    const std::vector<float>& c_pos,
    const std::vector<float>& c_rot,
    Eigen::Quaternionf& out_point_camera,
    Eigen::Quaternionf& out_rot_pcamera){

  //Point position UnrealEngineSystem to OpenCV/CloudCompare
  Eigen::Quaternionf qp_pos(0, p_pos[1], -p_pos[2], p_pos[0]);
  //Camera position
  Eigen::Quaternionf qc_pos(0, c_pos[1], -c_pos[2], c_pos[0]);

  //Point rotation quaternion
  Eigen::Quaternionf qp_rot = euler2quaternionYXZ(p_rot);
  //Camera rotation quaternion
  Eigen::Quaternionf qc_rot = euler2quaternionYXZ(c_rot);

  Eigen::Quaternionf rot_point  = qc_rot.conjugate() * qp_pos * qc_rot;
  Eigen::Quaternionf rot_camera = qc_rot.conjugate() * qc_pos * qc_rot;

  //q.w()//q.vec()
  out_rot_pcamera = qc_rot.conjugate() * qp_rot * qc_rot;
  auto vec = rot_point.vec() - rot_camera.vec();
  out_point_camera = Eigen::Quaternionf(0, vec.x(), vec.y(), vec.z());
}

void transform(const Eigen::Quaternionf& in_pframe,
    const Eigen::Quaternionf& in_rframe,
    const Eigen::Quaternionf& out_pframe,
    const Eigen::Quaternionf& out_rframe,
    Eigen::Quaternionf& output_point,
    Eigen::Quaternionf& output_rotation){

  Eigen::Quaternionf out_conj = out_rframe.conjugate();
  Eigen::Quaternionf rotated_output  = out_conj * out_pframe * out_rframe;
  Eigen::Quaternionf rotated_input   = out_conj * in_pframe  * out_rframe;

  output_rotation = out_conj * in_rframe * out_rframe;
  auto vec = in_pframe.vec() - out_pframe.vec();
  output_point    = Eigen::Quaternionf(0, vec.x(), vec.y(), vec.z());
}

}

#endif
