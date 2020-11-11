#include <iostream>
#include <vector>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
//#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/region_growing.h>
//#include <Python.h>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <sstream>
#include "happly.h"
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <jsoncpp/json/json.h>
#include "utils/transformation.hpp"
#include "utils/hand_model.h"
//#include <iostream>
//#include <filesystem>
#include <typeinfo>

//static std::string ROOT="data/unrealhands/raw/";
#define PARAM_ROOT "root"
#define PARAM_HELP "help"
//#define PARAM_RELATIVE "relative"
#define PARAM_VOXEL_GRID_SIZE "voxel_size"
#define PARAM_CURVATURE_SMOOTHNESS "curvature_smoothness"
#define PARAM_MAX_SIZE "max_size"
enum GT{GT_ALL=0, GT_RELATIVE=1, GT_ABSOLUTE=2};
enum EULERAXIS{EX, EY, EZ, EXY, EXZ, EYZ, EXYZ};

bool parse_command_line_options(boost::program_options::variables_map & pVariablesMap, const int & pArgc, char ** pArgv){
  try{
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
      (PARAM_HELP, "Produce help message")
      (PARAM_ROOT, boost::program_options::value<std::string>()->default_value("../../../icvl/training/raw"), "Root directory")
      (PARAM_VOXEL_GRID_SIZE, boost::program_options::value<float>()->default_value(0.05), "Voxel grid size")
      (PARAM_CURVATURE_SMOOTHNESS, boost::program_options::value<float>()->default_value(7.), "Curvature smoothness value")
      //(PARAM_RELATIVE, boost::program_options::value<bool>()->default_value(false), "Relative or absolute ground truth generation.");
      (PARAM_MAX_SIZE, boost::program_options::value<int>()->default_value(1000), "Clouds sampled above this number are deleted.");
    boost::program_options::store(boost::program_options::parse_command_line(pArgc, pArgv, desc), pVariablesMap);
    if (pVariablesMap.count(PARAM_HELP)){
      std::cout << desc << "\n";
      return true;
    }
  }catch(std::exception & e){
    std::cout << "ERROR" << std::endl;
    return true;
  }

  return false;
}

void downsample(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& downsampled, float grid_size){
  pcl::VoxelGrid<pcl::PointXYZRGB> sor;
  sor.setInputCloud (cloud);
  sor.setLeafSize (grid_size, grid_size, grid_size);
  sor.filter (*downsampled);
}

void writeHaplyFromPCL(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
    const std::vector<float>& ground_truth,
    const std::vector<std::array<double, 3> >&  joints_position,
    const std::string& output_name ){

  int size = cloud->points.size();
  std::vector<float> x(size),y(size),z(size);

  for(int i=0; i<size; ++i){
    x[i]=cloud->points[i].x;
    y[i]=cloud->points[i].y;
    z[i]=cloud->points[i].z;
  }

  happly::PLYData plyOut;
  plyOut.addElement("vertex", size);
  plyOut.getElement("vertex").addProperty<float>("x", x);
  plyOut.getElement("vertex").addProperty<float>("y", y);
  plyOut.getElement("vertex").addProperty<float>("z", z);

  plyOut.addElement("gt", ground_truth.size());
  plyOut.getElement("gt").addProperty<float>("gt", ground_truth);
  plyOut.addElement("jointsp", joints_position.size());

  size = joints_position.size();
  std::vector<float> xjp(size), yjp(size), zjp(size);
  for(size_t i =0; i<joints_position.size() ; ++i){
    xjp[i] = joints_position[i][0];
    yjp[i] = joints_position[i][1];
    zjp[i] = joints_position[i][2];
  }
  plyOut.getElement("jointsp").addProperty("x", xjp);
  plyOut.getElement("jointsp").addProperty("y", yjp);
  plyOut.getElement("jointsp").addProperty("z", zjp);

  plyOut.write(output_name, happly::DataFormat::Binary);
}

int main (int argc, char** argv){
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

  boost::program_options::variables_map variablesMap;
  if (parse_command_line_options(variablesMap, argc, argv))
    return 1;

  //Reading files ply
  std::string root = variablesMap[PARAM_ROOT].as<std::string>();
  if(!boost::filesystem::is_directory(root)){
    std::cout << "Directory " << root << " doesn't exists." << std::endl;
    return 1;
  }

  //bool relative = variablesMap[PARAM_RELATIVE].as<bool>();
  float grid_size = variablesMap[PARAM_VOXEL_GRID_SIZE].as<float>();
  float curvature_smoothness = variablesMap[PARAM_CURVATURE_SMOOTHNESS].as<float>();
  int max_size = variablesMap[PARAM_MAX_SIZE].as<int>();

  //for (const auto & entry : boost::make_iterator_range(boost::filesystem::directory_iterator(root), {})){
    std::string in_dir = root + "/cloud";
    std::string out_dir = root + "/cloud_sampled";
    //std::cout << in_dir << "\n" << out_dir << std::endl;
    std::cout << out_dir << std::endl;
    //Check cloud_sampled dir
    if(!boost::filesystem::is_directory(out_dir))
      boost::filesystem::create_directory(out_dir);

    std::vector<boost::filesystem::directory_entry> v;
    auto begin = boost::filesystem::directory_iterator(in_dir);
    auto end = boost::filesystem::directory_iterator();
    copy(begin, end, back_inserter(v));

      //std::cout << cin_dir << std::endl;

    #pragma omp parallel for
    for(int i=0; i<v.size(); ++i){
    //for(auto it=v.begin(); it != v.end(); ++it){
      const auto cloud_path = v[i];
      std::string filename = cloud_path.path().filename().string();
      std::string path = in_dir + "/" + filename;
      std::string output_name = out_dir + "/" + filename;
      std::cout << output_name << std::endl;

      std::cout << path << std::endl;
      int max = 0;
      if(!boost::filesystem::exists(output_name)){
        //Read ply. happly
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZRGB>());

        pcl::io::loadPLYFile(path, *cloud);
        happly::PLYData cloudply(path);

        downsample(cloud, downsampled, grid_size);
        //Write ply Happ TODO update to readed gt
        std::vector<float> gt = cloudply.getElement("gt").getProperty<float>("gt");
        std::vector<std::array<double, 3> > joints_position = cloudply.getVertexPositions("jointsp");

        //Filter clouds max sized
        if (downsampled->points.size() > max){
          max = downsampled->points.size();
          std::cout << max << std::endl;
        }
        if(downsampled->points.size() <= max_size){
          writeHaplyFromPCL(downsampled, gt, joints_position, output_name);
        }
      }
      std::cout << max << std::endl;
//    }
    }

  return 0;
}
