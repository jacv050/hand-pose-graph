#include <fstream>
#include <iostream>
#include <experimental/filesystem>

#include<pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/octree/octree_search.h>

const int kK = 3;
const std::string kPath = "/home/agarcia/Workspace/3dpl/2d-3d-segcn/data/nyudv2/raw/cloud";

struct Edge
{
	int m_src;
	int m_dst;
};

void write_ply(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pCloud, const std::vector<Edge> & crEdges, std::string filename)
{
	ofstream f_ (filename);

	if (f_.is_open())
	{
		f_ << "ply\n";
		f_ << "format ascii 1.0\n";
  }
  else
		std::cout << "Unable to open file";
}

int main()
{
	std::string path(kPath);
	std::string ext(".ply");

	std::vector<std::string> paths_;

	for(auto& p: std::experimental::filesystem::recursive_directory_iterator(path))
	{
		if(p.path().extension() == ext)
		{
			std::cout << p.path() << "\n";
			paths_.push_back(p.path());
		}
	}

	pcl::PLYReader reader_;

	for(std::string p : paths_)
	{
		std::cout << "Reading cloud " << p << "\n";

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ (new pcl::PointCloud<pcl::PointXYZRGB>);
		std::vector<Edge> edges_;

		reader_.read(p, *cloud_);

		//pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> tree_(128.0f);
		pcl::KdTreeFLANN<pcl::PointXYZRGB> tree_;
		tree_.setInputCloud(cloud_);
		//tree_.addPointsFromInputCloud();

		for (unsigned int i = 0; i < cloud_->size(); ++i)
		{
			std::vector<int> point_idx_(kK + 1);
			std::vector<float> point_distance_(kK + 1);

			pcl::PointXYZRGB search_point_;
			search_point_.x = cloud_->points[i].x;
			search_point_.y = cloud_->points[i].y;
			search_point_.z = cloud_->points[i].z;

			tree_.nearestKSearch(search_point_, kK + 1, point_idx_, point_distance_);

			for (size_t k = 1; k < point_idx_.size(); ++k)
			{
				Edge e1_;
				e1_.m_src = i;
				e1_.m_dst = point_idx_[k];
				Edge e2_;
				e2_.m_src = e1_.m_dst;
				e2_.m_dst = e1_.m_src;
				edges_.push_back(e1_);
				edges_.push_back(e2_);
			}
		}
	}

	return 0;
}
