#ifndef __transformation__
#define __transformation__

#include <Eigen/Geometry>
#include <jsoncpp/json/json.h>

namespace qutils{

enum RAXIS{RXYZ, RXZY, RYXZ, RYZX, RZXY, RZYX};

Eigen::Quaterniond json2pquat(const Json::Value& p){
  return Eigen::Quaterniond(0, p[0].asFloat(), p[1].asFloat(), p[2].asFloat());
}

Eigen::Quaterniond json2rquat(const Json::Value& p){
  return Eigen::Quaterniond(p[0].asFloat(), p[1].asFloat(), p[2].asFloat(), p[3].asFloat());
}

Eigen::Quaterniond euler2quaternionYXZ(const std::vector<float>& rot){
  return Eigen::AngleAxisd(rot[1], Eigen::Vector3d::UnitY())
             * Eigen::AngleAxisd(rot[2], Eigen::Vector3d::UnitX())
             * Eigen::AngleAxisd(rot[0], Eigen::Vector3d::UnitZ());
}

void world2camera(const std::vector<float>& p_pos,
    const std::vector<float>& p_rot,
    const std::vector<float>& c_pos,
    const std::vector<float>& c_rot,
    Eigen::Quaterniond& out_point_camera,
    Eigen::Quaterniond& out_rot_pcamera){

  //Point position UnrealEngineSystem to OpenCV/CloudCompare
  Eigen::Quaterniond qp_pos(0, p_pos[1], -p_pos[2], p_pos[0]);
  //Camera position
  Eigen::Quaterniond qc_pos(0, c_pos[1], -c_pos[2], c_pos[0]);

  //Point rotation quaternion
  Eigen::Quaterniond qp_rot = euler2quaternionYXZ(p_rot);
  //Camera rotation quaternion
  Eigen::Quaterniond qc_rot = euler2quaternionYXZ(c_rot);

  Eigen::Quaterniond rot_point  = qc_rot.conjugate() * qp_pos * qc_rot;
  Eigen::Quaterniond rot_camera = qc_rot.conjugate() * qc_pos * qc_rot;

  //q.w()//q.vec()
  out_rot_pcamera = qc_rot.conjugate() * qp_rot * qc_rot;
  auto vec = rot_point.vec() - rot_camera.vec();
  out_point_camera = Eigen::Quaterniond(0, vec.x(), vec.y(), vec.z());
}

//frame_pose, object_pose
void transform(
    const Eigen::Quaterniond& out_pframe,
    const Eigen::Quaterniond& out_rframe,
    const Eigen::Quaterniond& in_pframe,
    const Eigen::Quaterniond& in_rframe,
    Eigen::Quaterniond& output_point,
    Eigen::Quaterniond& output_rotation){

  Eigen::Quaterniond out_conj = out_rframe.conjugate();
  Eigen::Quaterniond rotated_output  = out_conj * out_pframe * out_rframe;
  Eigen::Quaterniond rotated_input   = out_conj * in_pframe  * out_rframe;

  auto vec = rotated_input.vec() - rotated_output.vec();
  output_point    = Eigen::Quaterniond(0, vec.x(), vec.y(), vec.z());
  output_rotation = in_rframe * out_conj;
}

//The order corresponds with RAXIS type
void rot3(double r11, double r12, double r21, double r31, double r32, std::vector<double>& res){
  res[0] = atan2( r11, r12 );
  res[1] = asin ( r21 );
  res[2] = atan2( r31, r32 );
}

void quaternion2euler(const Eigen::Quaterniond& q, std::vector<double>& res, RAXIS r){
  switch(r){
    case RXYZ:
      rot3(-2*(q.y()*q.z() - q.w()*q.x()),
          q.w()*q.w() - q.x()*q.x() - q.y()*q.y() + q.z()*q.z(),
          2*(q.x()*q.z() + q.w()*q.y()),
          -2*(q.x()*q.y() - q.w()*q.z()),
          q.w()*q.w() + q.x()*q.x() - q.y()*q.y() - q.z()*q.z(),
          res);
      break;
    case RXZY:
      rot3(2*(q.y()*q.z() + q.w()*q.x()),
          q.w()*q.w() - q.x()*q.x() + q.y()*q.y() - q.z()*q.z(),
          -2*(q.x()*q.y() - q.w()*q.z()),
          2*(q.x()*q.z() + q.w()*q.y()),
          q.w()*q.w() + q.x()*q.x() - q.y()*q.y() - q.z()*q.z(),
          res);
      break;
    case RYXZ:
      rot3(2*(q.x()*q.z() + q.w()*q.y()),
          q.w()*q.w() - q.x()*q.x() - q.y()*q.y() + q.z()*q.z(),
          -2*(q.y()*q.z() - q.w()*q.x()),
          2*(q.x()*q.y() + q.w()*q.z()),
          q.w()*q.w() - q.x()*q.x() + q.y()*q.y() - q.z()*q.z(),
          res);
    case RYZX:
     rot3(-2*(q.x()*q.z() - q.w()*q.y()),
         q.w()*q.w() + q.x()*q.x() - q.y()*q.y() - q.z()*q.z(),
         2*(q.x()*q.y() + q.w()*q.z()),
         -2*(q.y()*q.z() - q.w()*q.x()),
         q.w()*q.w() - q.x()*q.x() + q.y()*q.y() - q.z()*q.z(),
         res);
    case RZXY:
      rot3(-2*(q.x()*q.y() - q.w()*q.z()),
          q.w()*q.w() - q.x()*q.x() + q.y()*q.y() - q.z()*q.z(),
          2*(q.y()*q.z() + q.w()*q.x()),
          -2*(q.x()*q.z() - q.w()*q.y()),
          q.w()*q.w() - q.x()*q.x() - q.y()*q.y() + q.z()*q.z(),
          res);
    case RZYX:
      rot3(2*(q.x()*q.y() + q.w()*q.z()),
          q.w()*q.w() + q.x()*q.x() - q.y()*q.y() - q.z()*q.z(),
          -2*(q.x()*q.z() - q.w()*q.y()),
          2*(q.y()*q.z() + q.w()*q.x()),
          q.w()*q.w() - q.x()*q.x() - q.y()*q.y() + q.z()*q.z(),
          res);
  }
}

void print(const std::vector<double>& v){
  for(int i=0; i<v.size(); ++i){
    std::cout << v[i] << " ";
  }
  std::cout << std::endl;
}

}

#endif
