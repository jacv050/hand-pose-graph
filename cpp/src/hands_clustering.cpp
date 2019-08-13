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
//#include <iostream>
//#include <filesystem>

//static std::string ROOT="data/unrealhands/raw/";
#define PARAM_ROOT "root"
#define PARAM_HELP "help"
//#define PARAM_RELATIVE "relative"
#define PARAM_GROUND_TRUTH "ground_truth"
#define PARAM_VOXEL_GRID_SIZE "voxel_size"
#define PARAM_CURVATURE_SMOOTHNESS "curvature_smoothness"
enum GT{GT_ALL=0, GT_RELATIVE=1, GT_ABSOLUTE=2};

bool parse_command_line_options(boost::program_options::variables_map & pVariablesMap, const int & pArgc, char ** pArgv){
  try{
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
      (PARAM_HELP, "Produce help message")
      (PARAM_ROOT, boost::program_options::value<std::string>()->default_value("../../data/unrealhands/raw"), "Root directory")
      (PARAM_VOXEL_GRID_SIZE, boost::program_options::value<float>()->default_value(0.01), "Voxel grid size")
      (PARAM_CURVATURE_SMOOTHNESS, boost::program_options::value<float>()->default_value(7.), "Curvature smoothness value")
      //(PARAM_RELATIVE, boost::program_options::value<bool>()->default_value(false), "Relative or absolute ground truth generation.");
      (PARAM_GROUND_TRUTH, boost::program_options::value<uint8_t>()->default_value(0), "(Default)ALL=0, RELATIVE=1, ABSOLUTE=2");
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

void read_ground_truth_json(const std::string& filename,
      //void (*operation)(const Json::Value&, const Json::Value&, std::vector<float>&),
      std::vector< void (*)(bool, const Json::Value&, std::vector<float>& ) > operations,
      std::vector< std::vector<float> >& left,
      std::vector< std::vector<float> >& right){
  //std::cout << filename << std::endl;
  std::ifstream ground_truth(filename, std::ifstream::binary);
  Json::Reader reader;
  Json::Value obj;

  reader.parse(ground_truth, obj);
  const Json::Value& left_hand = obj["left_hand"];
  const Json::Value& right_hand = obj["right_hand"];

  //operations order -> relative, absolute, ...
  for(int k=0; k<operations.size(); ++k){
    std::vector<float> l, r;
    void (*operation)(bool, const Json::Value&, std::vector<float>&) = operations[k];
    //Always root at begining
    for(int i=0; i<3; ++i){
      l.push_back(left_hand[0][i].asFloat());
      r.push_back(right_hand[0][i].asFloat());
    }

    for (int i=1; i<left_hand.size(); ++i){
      const Json::Value& lfinger = left_hand[i];
      const Json::Value& rfinger = right_hand[i];

      operation(true, lfinger[0], l);
      operation(true, rfinger[0], r);

      operation(false, lfinger[1], l);
      operation(false, rfinger[1], r);

      operation(false, lfinger[2], l);
      operation(false, rfinger[2], r);
    }

    left.push_back(l);
    right.push_back(r);
  }
}

//TODO first parameter doesn't used, valorate to delete.
void absolute(bool init_finger, const Json::Value& p2, std::vector<float>& o){
  int size = o.size();
  if(init_finger){
    o.push_back(o[0]+p2[0].asFloat());
    o.push_back(o[1]+p2[1].asFloat());
    o.push_back(o[2]+p2[2].asFloat());
    //o.push_back(p1[0].asFloat()+p2[0].asFloat());
    //o.push_back(p1[1].asFloat()+p2[1].asFloat());
    //o.push_back(p1[2].asFloat()+p2[2].asFloat());
  }else{
    o.push_back(o[size-3]+p2[0].asFloat());
    o.push_back(o[size-2]+p2[1].asFloat());
    o.push_back(o[size-1]+p2[2].asFloat());
  }
}

//TODO first parameter doesn't used, valorate to delete.
void relative(bool init_finger, const Json::Value& p2, std::vector<float>& o){
  o.push_back(p2[0].asFloat());
  o.push_back(p2[1].asFloat());
  o.push_back(p2[2].asFloat());
}

void generate_ground_truth(const std::string& filename,
      GT mode, std::vector< std::vector<float> >& left_hand,
      std::vector< std::vector<float> >& right_hand){
  //void (*operation)(const Json::Value&, const Json::Value&, std::vector<float>&);
  std::vector< void (*)(bool, const Json::Value&, std::vector<float>& ) > operations;

  switch(mode){
    case GT_RELATIVE:
      operations.push_back(relative);
      break;
    case GT_ABSOLUTE:
      operations.push_back(absolute);
      break;
    case GT_ALL:
    default:
      operations.push_back(relative);
      operations.push_back(absolute);
      break;
  }

  /*
  if(relative)
    operations.push_back([](const Json::Value& p1, const Json::Value& p2, std::vector<float>& o){
        o.push_back(p1[0].asFloat()+p2[0].asFloat());o.push_back(p1[1].asFloat()+p2[1].asFloat());o.push_back(p1[2].asFloat()+p2[2].asFloat());});
  else
    operations.push_back([](const Json::Value& p1, const Json::Value& p2, std::vector<float>& o){
        o.push_back(p2[0].asFloat());o.push_back(p2[1].asFloat());o.push_back(p2[2].asFloat());});
  */

  read_ground_truth_json(filename, operations, left_hand, right_hand);
}

void writeHaplyFromPCL(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
    GT mode,
    const std::vector< std::vector<float> >& ground_truth,
    const std::string& output_name ){

  std::vector<float> x,y,z;
  std::vector<uint8_t> r, g, b;

  int size = cloud->points.size();
  for(int i=0; i<size; ++i){
    x.push_back(cloud->points[i].x);
    y.push_back(cloud->points[i].y);
    z.push_back(cloud->points[i].z);
    r.push_back(cloud->points[i].r);
    g.push_back(cloud->points[i].g);
    b.push_back(cloud->points[i].b);
  }

  happly::PLYData plyOut;
  plyOut.addElement("vertex", size);
  plyOut.getElement("vertex").addProperty<float>("x", x);
  plyOut.getElement("vertex").addProperty<float>("y", y);
  plyOut.getElement("vertex").addProperty<float>("z", z);
  plyOut.getElement("vertex").addProperty<uint8_t>("red", r);
  plyOut.getElement("vertex").addProperty<uint8_t>("green", g);
  plyOut.getElement("vertex").addProperty<uint8_t>("blue", b);

  plyOut.addElement("gt", ground_truth[0].size());
  switch(mode){
    case GT_RELATIVE:
      plyOut.getElement("gt").addProperty<float>("relative", ground_truth[0]);
      break;
    case GT_ABSOLUTE:
      plyOut.getElement("gt").addProperty<float>("absolute", ground_truth[0]);
      break;
    case GT_ALL:
    default:
      plyOut.getElement("gt").addProperty<float>("relative", ground_truth[0]);
      plyOut.getElement("gt").addProperty<float>("absolute", ground_truth[1]);
      break;
  }

  if(mode != GT_RELATIVE){
    std::vector<float> jx, jy, jz;
    for(int i=0; i<ground_truth[1].size(); i+=3){
      jx.push_back(ground_truth[1][i]);
      jy.push_back(ground_truth[1][i+1]);
      jz.push_back(ground_truth[1][i+2]);
    }
    plyOut.addElement("joints", jx.size());
    plyOut.getElement("joints").addProperty<float>("x", jx);
    plyOut.getElement("joints").addProperty<float>("y", jy);
    plyOut.getElement("joints").addProperty<float>("z", jz);
  }

  plyOut.write(output_name, happly::DataFormat::Binary);
}

void generate_output_name(const std::string& name, const std::string& output_format, std::string& output_name){
  output_name = "";
  for(int i=0; i<name.length(); ++i){
    if(name[i] == '.'){
      output_name += output_format;
      return;
    }else
      output_name += name[i];
  }
}

void index_points_near_gt(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
      const pcl::search::Search<pcl::PointXYZRGB>::Ptr& search,
      const std::vector<float>& lground_truth,
      const std::vector<float>& rground_truth,
      std::vector<int>& lindices,
      std::vector<int>& rindices){
  //std::cout << lground_truth.size() << std::endl;
  //48 sized vector
  for(int i=0; i<lground_truth.size(); i+=3){
  //for(int i=0, j=0; i<lground_truth.size(); i+=3, ++j){
    std::vector<int> li, ri;
    std::vector<float> distances;

    pcl::PointXYZRGB lp(0,0,0);
    pcl::PointXYZRGB rp(0,0,0);

    lp.x = lground_truth[i];
    lp.y = lground_truth[i+1];
    lp.z = lground_truth[i+2];

    rp.x = rground_truth[i];
    rp.y = rground_truth[i+1];
    rp.z = rground_truth[i+2];

    search->nearestKSearch(lp, 1, li, distances);
    //distances.clear();
    search->nearestKSearch(rp, 1, ri, distances);
    lindices.push_back(li[0]);
    rindices.push_back(ri[0]);
    //std::cout << "GT: " << i << "\n";
    //std::cout << lground_truth[i] << " " << lground_truth[i+1] << " " << lground_truth[i+2] << std::endl;
  }
}

bool exists(std::vector<int> v, int n){
  for(int i=0;i<v.size(); ++i){
    if(v[i] == n)
      return true;
  }
  return false;
}

void segment_hand(pcl::PointIndices& cluster, const std::vector<int>& indices,
      pcl::RegionGrowing<pcl::PointXYZRGB, pcl::Normal>& reg,
      const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& in_cloud,
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr& out_cloud){
  for(int i=0; i<indices.size(); ++i){
    //std::cout << "Index: " << indices[i] << std::endl;
    if(cluster.indices.size() == 0){
      //pcl::PointXYZ p(lground_truth[i], lground_truth[i+1], lground_truth[i+2]);
      reg.getSegmentFromPoint(indices[i], cluster);
      //std::cout << cluster.indices.size() << std::endl;
    }else if(!exists(cluster.indices, indices[i])){
      pcl::PointIndices c;
      reg.getSegmentFromPoint(indices[i], c);
      cluster.indices.insert(cluster.indices.end(), c.indices.begin(), c.indices.end());
      //std::cout << cluster.indices.size() << std::endl;
    }
  }

  for (std::vector<int>::const_iterator pit = cluster.indices.begin (); pit != cluster.indices.end (); ++pit)
    out_cloud->points.push_back(in_cloud->points[*pit]);

  out_cloud->width = out_cloud->points.size();
  out_cloud->height = 1;
  out_cloud->is_dense = true;
}

void separete_hands(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr& left_hand,
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr& right_hand,
      const std::vector<float>& lground_truth,
      const std::vector<float>& rground_truth,
      float curvature_smoothness){
  pcl::search::Search<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
  pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> normal_estimator;
  normal_estimator.setSearchMethod (tree);
  normal_estimator.setInputCloud (cloud);
  normal_estimator.setKSearch (50);
  normal_estimator.compute (*normals);

  /*
  pcl::IndicesPtr indices (new std::vector <int>);
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud (cloud);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (0.0, 1.0);
  pass.filter (*indices);
  */

  pcl::RegionGrowing<pcl::PointXYZRGB, pcl::Normal> reg;
  reg.setMinClusterSize (1);
  reg.setMaxClusterSize (15000);
  reg.setSearchMethod (tree);
  reg.setNumberOfNeighbours (10);
  reg.setInputCloud (cloud);
  //reg.setIndices (indices);
  reg.setInputNormals (normals);
  //reg.setSmoothnessThreshold (3.0 / 180.0 * M_PI);
  reg.setSmoothnessThreshold (curvature_smoothness / 180.0 * M_PI);
  reg.setCurvatureThreshold (1.0);

  std::vector<int> lindices;
  std::vector<int> rindices;
  index_points_near_gt(cloud, tree, lground_truth, rground_truth, lindices, rindices);

  //std::vector<pcl::PointIndices> clusters;
  pcl::PointIndices lcluster, rcluster;
  //reg.extract(clusters);
  //std::cout << "Segment Left" << std::endl;
  segment_hand(lcluster, lindices, reg, cloud, left_hand);
  //std::cout << "Segment Right" << std::endl;
  segment_hand(rcluster, rindices, reg, cloud, right_hand);
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
  GT gt = (GT) variablesMap[PARAM_GROUND_TRUTH].as<uint8_t>();

  float grid_size = variablesMap[PARAM_VOXEL_GRID_SIZE].as<float>();
  float curvature_smoothness = variablesMap[PARAM_CURVATURE_SMOOTHNESS].as<float>();

  for (const auto & entry : boost::make_iterator_range(boost::filesystem::directory_iterator(root), {})){
    std::string in_dir = entry.path().string() + "/cloud";
    std::string out_dir = entry.path().string() + "/cloud_sampled";
    std::string joints = entry.path().string() + "/joints";
    //std::cout << in_dir << "\n" << out_dir << std::endl;
    for (const auto & centry : boost::make_iterator_range(boost::filesystem::directory_iterator(in_dir), {})){
      //std::ostringstream oss;
      //oss << centry;
      std::string camera = centry.path().filename().string();
      std::string cin_dir = in_dir + "/" + camera;
      std::string cout_dir = out_dir + "/" + camera;
      std::string cjoints = joints + "/" + camera;

      std::vector<boost::filesystem::directory_entry> v;
      auto begin = boost::filesystem::directory_iterator(cin_dir);
      auto end = boost::filesystem::directory_iterator();
      copy(begin, end, back_inserter(v));

      #pragma omp parallel for
      for(int i=0; i<v.size(); ++i){
      //for(auto it=v.begin(); it != v.end(); ++it){
        const auto cloud_path = v[i];
        std::string filename = cloud_path.path().filename().string();
        std::string path = cin_dir + "/" + filename;
        std::string fjoints = filename;

        std::string loutput_name, routput_name, filename_joints;
        generate_output_name(filename, "_lsampled.ply", loutput_name);
        generate_output_name(filename, "_rsampled.ply", routput_name);
        generate_output_name(fjoints, ".json", filename_joints);
        loutput_name = cout_dir + "/" + loutput_name;
        routput_name = cout_dir + "/" + routput_name;
        filename_joints = cjoints + "/" + filename_joints;

        std::cout << path << std::endl;
        if(!boost::filesystem::exists(loutput_name) || !boost::filesystem::exists(routput_name)){
        //Read ply. happly
        /****
        happly::PLYData plyIn(cloud);
        std::vector<float> x = plyIn.getElement("vertex").getProperty<float>("x");
        std::vector<float> y = plyIn.getElement("vertex").getProperty<float>("y");
        std::vector<float> z = plyIn.getElement("vertex").getProperty<float>("z");
        std::vector<uint8_t> r = plyIn.getElement("vertex").getProperty<uint8_t>("red");
        std::vector<uint8_t> g = plyIn.getElement("vertex").getProperty<uint8_t>("green");
        std::vector<uint8_t> b = plyIn.getElement("vertex").getProperty<uint8_t>("blue");
        ****/
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZRGB>());

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr left_hand(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr right_hand(new pcl::PointCloud<pcl::PointXYZRGB>());

        pcl::io::loadPLYFile(path, *cloud);
        downsample(cloud, downsampled, grid_size);
        //Write ply Happ
        std::vector< std::vector<float> > gt_left_hand, gt_right_hand;

        //gt -> enum ground truth mode
        generate_ground_truth(filename_joints, gt, gt_left_hand, gt_right_hand);
        //cluster cloud GROUND TRUTH 1 = ABSOLUTE
        separete_hands(downsampled, left_hand, right_hand, gt_left_hand[1], gt_right_hand[1], curvature_smoothness);

        writeHaplyFromPCL(left_hand, gt, gt_left_hand, loutput_name);
        writeHaplyFromPCL(right_hand, gt, gt_right_hand, routput_name);
      }
      }
      //}
    }
  }

  /*
  if ( pcl::io::loadPCDFile <pcl::PointXYZ> ("output_cloud_5.ply", *cloud) == -1)
  {
    std::cout << "Cloud reading failed." << std::endl;
    return (-1);
  }*/
  //#load cloud from python binded

  return 0;
}
