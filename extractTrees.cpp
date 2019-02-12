#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/flann.hpp"
#include "opencv2/highgui.hpp"
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <stdlib.h>

#include <opencv2/viz/vizcore.hpp>

#define numData 100000
float RADIUS[16] = {0.05, 0.1,
			0.2, 0.3, 0.4,
			0.5, 0.6, 0.8, 0.95,
			1.1, 1.3, 1.5, 2, 3, 4};

void read_pointcloud(cv::Mat &pointCloud, cv::Mat &GTlabel, char *fileName){
	std::string line;
	std::ifstream infile(fileName);
	for(int i=0;i<63;i++)
		std::getline(infile, line);			// Meta data
	int count = 0;
	while (std::getline(infile, line)){
		std::istringstream iss(line);
		float a,b,c;
		int d;
		if (!(iss >>a>>b>>c>>d))
			break; 		 			// error
		pointCloud.at<float>(count, 0) = a; pointCloud.at<float>(count, 1) =b; pointCloud.at<float>(count, 2) = c;
		GTlabel.at<int>(count,0) = d;
		count++;
		// put in point cloud;
	}
}

float rgb2float(int r,int g, int b){
	//std::cout<<label<<"\n";
	return float(r<<16 | g<<8 | b);
}

void write_pcl_labels(cv::Mat pointCloud, cv::Mat labels,const char* fileName){
	std::ofstream ofile (fileName, std::ios::out);
	ofile<<"VERSION .7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH ";
	ofile<<pointCloud.rows<<"\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS ";
	ofile<<pointCloud.rows<<"\nDATA ascii\n";
	std::vector<int> sum;
	int count = 0;
	sum.push_back(0);
	sum.push_back(0);
	sum.push_back(0);
	for(int i = 0; i < pointCloud.rows; i++){
		sum[labels.at<int>(i,0)] += 1;
		ofile<<pointCloud.at<float>(i,0)<<' '<<pointCloud.at<float>(i,1)<<' '<<pointCloud.at<float>(i,2)<<' '<<rgb2float(labels.at<int>(i,0)*75, 50, 150)<<'\n';
	}
	
	for(int i = 0; i < sum.size(); i++)
		std::cout<<sum[i]<<"   ";
	ofile.close();
}

void write_pcl_labels_from_set(cv::Mat pointCloud, std::set<int> trees, char* fileName){
	std::ofstream ofile (fileName, std::ios::out);
	ofile<<"VERSION .7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH ";
	ofile<<trees.size()<<"\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS ";
	ofile<<trees.size()<<"\nDATA ascii\n";
	std::vector<int> sum;
	int count = 0;
	sum.push_back(0);
	sum.push_back(0);
	sum.push_back(0);
	for (std::set<int>::iterator i = trees.begin(); i != trees.end(); i++) {
		ofile<<pointCloud.at<float>(*i,0)<<' '<<pointCloud.at<float>(*i,1)<<' '<<pointCloud.at<float>(*i,2)<<' '<<rgb2float(0,255, 0)<<'\n';
	}	
}

int max_index(cv::Vec3f scores){
	if(scores[0] > scores[1] and scores[0] > scores[2])
		return 0;
	else if(scores[1] > scores[0] and scores[1] > scores[2])
		return 1;
	else
		return 2;
}

cv::Mat extract_indices(cv::Mat pointCloud, std::vector<int> indices){
	cv::Mat cluster;
	for(int i = 0; i < indices.size(); i++){
		if(indices[i] == 0) continue;
		cluster.push_back(pointCloud.row(indices[i]));
	}
	return cluster;
}



float R[16] = { 0.25, 0.3, 0.4,										// Search Radius
		0.5, 0.7, 0.9, 1.2,
		1.4, 1.7, 1.8, 2.1,
		2.4, 2.8, 3.2, 3.5,4.0};

bool get_initial_labels(cv::Mat &pointCloud, cv::Mat &labels, cv::flann::Index &kdtree,float radius = 0.4){
	std::cout << "Performing single search to find closest data points\n";
        std::cout <<radius<<std::endl;
	for(int i = 0; i <pointCloud.rows; i++){
		std::vector<float> point, dists;
		std::vector<int> index;
		
		float minEntropy = 100000;
		for(int r = 0; r < 16; r++){
			kdtree.radiusSearch(pointCloud.row(i), index, dists, RADIUS[r],70, cv::flann::SearchParams(64));			// find nearest neighbours
			cv::Mat cluster = extract_indices(pointCloud, index);
			if(cluster.rows < 10){
                                labels.at<int>(i, 0)  = 2;
				continue;
			}
			cv::PCA pca_analysis(cluster, cv::Mat(), CV_PCA_DATA_AS_ROW);
			cv::Vec3f eigenvalues, dimenLabel;
			for(int j = 0; j < 3; j++)
				eigenvalues[j] = pca_analysis.eigenvalues.at<float>(0, j);					// linear, planar or volumetric
			cv::sqrt(eigenvalues, eigenvalues);
			dimenLabel[0] = (eigenvalues[0] - eigenvalues[1])/eigenvalues[0];
			dimenLabel[1] = (eigenvalues[1] - eigenvalues[2])/eigenvalues[0];;
			dimenLabel[2] = (eigenvalues[2])/eigenvalues[0];;
			cv::Vec3f log_eig;
			cv::log(dimenLabel, log_eig);
			float entropy = -dimenLabel[0]*log_eig[0] - dimenLabel[1]*log_eig[1] - dimenLabel[2]*log_eig[2];   //  minmize entropy for best radius
			if(entropy < minEntropy){
				//std::cout<<entropy<<"   "<<R[r]<<'\n';
				minEntropy = entropy;
				labels.at<int>(i, 0)  = max_index(dimenLabel);
			}
		}
	}
	std::cerr<<" done ";
        return true;
}

float C[3][3] = {{0.83, 0.08, 0.19}, {0.08, 0.91, 0.01}, {0.19,  0.01, 0.80}};            // label transition probablities

void fill_P(cv::Mat labels, float P[numData][3], float GT){					// load initial probablities
	for(int i = 0; i < labels.rows; i++){
		for(int k = 0; k < 3; k++){
			if(labels.at<int>(i,0) == k)
				P[i][k] = GT;
			else
				P[i][k] = (1 - GT)/2;
		}
	}
}

void label_point_cloud(float P[numData][3], cv::Mat &labels){				// label points from probablities
	for( int i = 0; i < labels.rows; i++){
		cv::Vec3f score;
		for(int k = 0; k < 3; k++){
			score[k] = P[i][k];
		}
		labels.at<int>(i,0) = max_index(score);
	}
}

double pdf(double x, double mu, double sigma){						//Gaussin Kernel
  //Constants
  static const double pi = 3.14159265; 
  return exp( -1 * (x - mu) * (x - mu) / (2 * sigma * sigma)) / (sigma * sqrt(2 * pi));
}

void apply_probablistic_relaxation(cv::Mat &pointCloud, cv::Mat &labels, cv::flann::Index &kdtree, float GT, int iterations){
	float P[numData][3];
	fill_P(labels, P, GT);
	for(int t = 0; t < iterations; t++){
		for(int i = 0; i < pointCloud.rows; i++){
			std::vector<float> dists;
			std::vector<int> index;
			kdtree.radiusSearch(pointCloud.row(i), index, dists, 1, 50, cv::flann::SearchParams(64));		// find nearest neighbours
			//cv::Mat cluster = extract_indices(pointCloud, index);
			float deltaP[3] = {0 ,0 ,0};
			for(int k = 0; k < 3; k++){
				float sumO = 0;
				for(int c = 0; c < index.size(); c++){
					if(index[c] == 0) continue;
					float sumI = 0;
					for(int l = 0; l < 3; l++)
						sumI += C[k][l]*P[index[c]][l];				
					sumO += pdf(dists[c], 0, 0.5)*sumI;
				}
				deltaP[k] = sumO;
			}
			float Q[3] = {0,0,0}, sumQ = 0;
			for(int k = 0; k < 3; k++){
				Q[k] = P[i][k]*(1 + deltaP[k]);
				sumQ += Q[k];
			}	
			for(int k = 0; k < 3; k++)
				P[i][k] = Q[k]/sumQ;
		}
	
		label_point_cloud(P, labels);
	}
}

int extract_points(cv::Mat &pointCloud, cv::Mat &labels, cv::Mat &GTlabels, cv::flann::Index &kdtree, int &total_points, char* fileName){
	int count = 0;
	std::set<int> trees;
	for(int i = 0; i < pointCloud.rows; i++){
		std::vector<float> dists;
		std::vector<int> index;
		if(labels.at<int>(i,0) == 1 or labels.at<int>(i,0) == 0) 
			continue;
		kdtree.radiusSearch(pointCloud.row(i), index, dists, 1.5, 200, cv::flann::SearchParams(64));		// find nearest neighbours
		for(int c = 0; c < index.size(); c++){
			if(index[c] == 0 or (labels.at<int>(index[c], 0) == 1 and dists[c] > 1) or (labels.at<int>(index[c], 0) == 0 and  dists[c] > 1) or (labels.at<int>(index[c], 0) == 2 and pointCloud.at<float>(index[c],2) < 1.5) )  continue;
			else{
				std::pair<std::set<int>::iterator,bool> result;
				result = trees.insert(index[c]);
				if(result.second and GTlabels.at<int>(index[c], 0) == 1300)
					count++;
			}
		}
	}
	total_points = trees.size();
	write_pcl_labels_from_set(pointCloud, trees, fileName);	
	return count;
}

void knn_classify(cv::Mat &pointCloud, cv::Mat &labels, cv::flann::Index &kdtree, int iterations){
	for(int t = 0; t < iterations; t++){
		cv::Mat new_labels(numData, 1, CV_32S);
		for(int i = 0; i < pointCloud.rows; i++){
			//std::cerr<<i<<"  in\n";
			std::vector<float> dists;
			std::vector<int> index;
			kdtree.radiusSearch(pointCloud.row(i), index, dists, 0.15, 200, cv::flann::SearchParams(64));		// find nearest neighbours
			cv::Vec3f score;
			for(int c = 0; c < index.size(); c++){
				if(index[c] == 0) continue;
				score[labels.at<int>(index[c],0)]++;
			}
			int label =  max_index(score);
			if(labels.at<int>(i, 0) == 0)continue;
			else labels.at<int>(i, 0) = label;
		}
	}
}

float treeRadius[11] = { 1, 1.3, 1.7, 1.9, 2.1, 2.3, 2.6, 2.9, 3.2, 3.5, 4};

void ransac(cv::Mat pointCloud, cv::Mat labels, cv::flann::Index kdtree){
	std::vector<float> scores, heights, radii;
	for(int i = 0; i < pointCloud.rows; i++){
		float maxScore = -100;
		float radius = 0, height = 0;
		for(int k = 0; k < 11; k++){
			float score = 0;
			int count = 0;
			std::vector<float> dists;
			std::vector<int> index;
			kdtree.radiusSearch(pointCloud.row(i), index, dists, treeRadius[k], 300, cv::flann::SearchParams(64));		// find nearest neighbours
			for(int j = 0; j < index.size(); j++){
				if(index[j] == 0) continue;
				count++;
				if(labels.at<int>(j,0) == 1)
					score--;
				else
					score++;
				height += pointCloud.at<float>(index[j],2);
			}
			score = score/count;
			if(score > maxScore){
				maxScore = score;
				height = height/count;
				radius = treeRadius[k];
			}
		}
		scores.push_back(maxScore);
		heights.push_back(height);
		radii.push_back(radius);
	}
	for(int i = 0; i < pointCloud.rows; i++){
		if(heights[i] > 1.8 and radii[i] > 1.4 and scores[i] >=0.7 )
			std::cout<<"this is a tree\n";
	}
}

void evalute_result(int true_positive,int total_points, cv::Mat GTlabels){
	int count = 0;
	for(int i = 0; i < GTlabels.rows; i++)
		if(GTlabels.at<int>(i,0) == 1300)
			count++;
	std::cout<<"\ntrue_positive :: "<<true_positive;
	std::cout<<"\nfalse_positive:: "<<total_points - true_positive;
	std::cout<<"\naccuracy      :: "<<float(true_positive)/total_points;
	std::cout<<"\nrecall        :: "<<float(true_positive)/count;
	
}
int main(int argc, char** argv){
	if(argc <4){
		std::cout<<"./extarctTrees data_file_with_labels label_output.pcd extracted_trees.pcd\n\n";
		exit(0);
	}
	char *fileName = *(argv + 1);
	cv::Mat pointCloud(numData, 3, CV_32F);
	
	cv::Mat GTlabels(numData, 1, CV_32S);
	cv::Mat labels(numData, 1, CV_32S);
	//fill data;	
	read_pointcloud(pointCloud, GTlabels, fileName);
	
	std::cout<<pointCloud.rows<<" pointCloud : \n";
	
	cv::flann::KDTreeIndexParams indexParams(5);								// Create the Index for kdTree
	cv::flann::Index kdtree(pointCloud, indexParams,cvflann::FLANN_DIST_EUCLIDEAN);				// Create KD Tree for search

	std::cerr<<"getting initial labels";
	get_initial_labels(pointCloud,labels,kdtree, 0.3);							//get distribution labels
	
	int iterations = 3;

	std::cerr<<"relaxing labels";
	apply_probablistic_relaxation(pointCloud, labels, kdtree, 0.6, iterations);				//probablistic smoothing

	std::cerr<<"labels relaxed";
	
	//write_pcl_labels(pointCloud, labels, "out1.pcd");
	//knn_classify(pointCloud, labels, kdtree, 3);
	
	write_pcl_labels(pointCloud, labels, *(argv+2));							// file with initial labels	
	int total_points = 0;
	int true_positive = extract_points(pointCloud, labels, GTlabels, kdtree, total_points, *(argv + 3));			// Trees extracted by spatial filtering
	evalute_result(true_positive, total_points, GTlabels);
//	ransac(pointCloud, labels, kdtree);
	return 0;
}
