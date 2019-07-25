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
#include <>
//#include <iostream>
//#include <filesystem>

//static std::string ROOT="data/unrealhands/raw/";
#define PARAM_ROOT "root"
#define PARAM_HELP "help"

bool parse_command_line_options(boost::program_options::variables_map & pVariablesMap, const int & pArgc, char ** pArgv){
  try{
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
      (PARAM_HELP, "Produce help message")
      (PARAM_ROOT, boost::program_options::value<std::string>()->default_value("../../data/unrealhands/raw"), "Root directory");
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

void downsample(){

}

int main (int argc, char** argv){
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

  boost::program_options::variables_map variablesMap;
  if (parse_command_line_options(variablesMap, argc, argv))
    return 1;

  //Reading files ply
  std::string root = variablesMap[PARAM_ROOT].as<std::string>();
  if(!boost::filesystem::is_directory(root)){
    std::cout << "Directory " << root << " doesn't exists." << std::endl;
    return 1;
  }

  for (const auto & entry : boost::make_iterator_range(boost::filesystem::directory_iterator(root), {})){
    std::string in_dir = entry.path().string() + "/cloud";
    std::string out_dir = entry.path().string() + "/cloud_sampled";
    //std::cout << in_dir << "\n" << out_dir << std::endl;
    for (const auto & centry : boost::make_iterator_range(boost::filesystem::directory_iterator(in_dir), {})){
      //std::ostringstream oss;
      //oss << centry;
      std::string camera = centry.path().filename().string();
      std::string cin_dir = in_dir + "/" + camera;
      std::string cout_dir = out_dir + "/" + camera;
      for (const auto & cloud_path : boost::make_iterator_range(boost::filesystem::directory_iterator(cin_dir), {})){
        std::string cloud = cloud_path.path().string();
        
      }
    }
  }

  /*
  if ( pcl::io::loadPCDFile <pcl::PointXYZ> ("output_cloud_5.ply", *cloud) == -1)
  {
    std::cout << "Cloud reading failed." << std::endl;
    return (-1);
  }*/
  //#load cloud from python binded

  return -1;

  pcl::search::Search<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
  normal_estimator.setSearchMethod (tree);
  normal_estimator.setInputCloud (cloud);
  normal_estimator.setKSearch (50);
  normal_estimator.compute (*normals);

  pcl::IndicesPtr indices (new std::vector <int>);
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud (cloud);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (0.0, 1.0);
  pass.filter (*indices);

  pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
  reg.setMinClusterSize (50);
  reg.setMaxClusterSize (1000000);
  reg.setSearchMethod (tree);
  reg.setNumberOfNeighbours (30);
  reg.setInputCloud (cloud);
  //reg.setIndices (indices);
  reg.setInputNormals (normals);
  reg.setSmoothnessThreshold (3.0 / 180.0 * M_PI);
  reg.setCurvatureThreshold (1.0);

  std::vector<pcl::PointIndices> clusters;
  reg.extract(clusters);

  

  return 0;
}
