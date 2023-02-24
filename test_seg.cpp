#include "seg.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <string>
#include <set>

// #include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

std::array<uchar, 3> getColor(uchar id)
{
  std::array<uchar, 3> ret;
  switch (id % 12)
  {
    case 0:
      return std::array<uchar, 3>{0, 0, 0}; // background
    case 1:
      return std::array<uchar, 3>{255, 0, 0}; // wall
    case 2:
      return std::array<uchar, 3>{0, 0, 255}; // building
    case 3:
      return std::array<uchar, 3>{255, 255, 255}; // sky
    case 4:
      return std::array<uchar, 3>{255, 255, 0}; // floor
    case 5:
      return std::array<uchar, 3>{0, 255, 255}; // ceiling
    case 6:
      return std::array<uchar, 3>{140, 140, 140}; //highway
    case 7:
      return std::array<uchar, 3>{255, 0, 255}; //sidewalk
    case 8:
      return std::array<uchar, 3>{0, 255, 0}; //grass
    case 9:
      return std::array<uchar, 3>{230, 128, 30}; // person
    case 10:
      return std::array<uchar, 3>{128, 0, 0}; // door
    case 11:
      return std::array<uchar, 3>{0, 0, 128}; // table
  }
  return ret;
}

void segmentation(const char *root_)
{
  InitPlugin();

  const std::string root = root_;
  const std::string &cam0_dir = root + std::string("/images/device_1/");
  const std::string &cam1_dir = root + std::string("/images/device_2/");
  const std::string &save0_dir = root + std::string("/labels/device_1/");
  const std::string &save1_dir = root + std::string("/labels/device_2/");
  const std::string &visual_dir = root + std::string("/labels/visual/");
  const std::string &config_file = root + std::string("/img_pose.txt");

  std::ifstream fts;
  fts.open(config_file);
  if (!fts.is_open())
  {
    std::cerr << "img_pose path " << config_file.c_str() << "false\n";
    return ;
  }

  // if (!boost::filesystem::exists(save0_dir))
  //   boost::filesystem::create_directories(save0_dir);
  // if (!boost::filesystem::exists(save1_dir))
  //   boost::filesystem::create_directories(save1_dir);

  float n;
  int w = 384, h = 288, m;
  int ow = w, oh = h;
  int iw = 640, ih = 480;
  int nbytes = w * h;
  int start_id = 1057, end_id = 1542;

  std::string name;
  cv::Mat visual_img = cv::Mat::zeros(h, w, CV_8UC3);

  while (!fts.eof())
  {
    fts >> n >> n >> n >> n >> n >> n >> n >> m >> name;
    if (name.empty())
      break;
    cv::Mat left, right, left_out, right_out;
    left = cv::imread((cam0_dir + name).c_str(), 0);
    right = cv::imread((cam1_dir + name).c_str(), 0);
    if( left.empty() || right.empty())
      continue;
    cv::resize(left, left, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
    cv::resize(right, right, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);

    assert(left.type() == 0 && right.type() == 0);

    unsigned char *input_l = (unsigned char *)malloc(nbytes);
    unsigned char *input_r = (unsigned char *)malloc(nbytes);
    memcpy(input_l, left.ptr<unsigned char>(0), nbytes);
    memcpy(input_r, right.ptr<unsigned char>(0), nbytes);

    unsigned char *output_l = (unsigned char *)malloc(nbytes);
    unsigned char *output_r = (unsigned char *)malloc(nbytes);
    bool res_l = InferenceSeg(input_l, w, h, output_l, ow, oh);
    bool res_r = InferenceSeg(input_r, w, h, output_r, ow, oh);

    if (!res_l || !res_r)
    {
      std::cerr << "segmantation fail for " << name.c_str() << std::endl;
      return ;
    }

    memcpy(left.ptr<unsigned char>(0), output_l, nbytes);
    memcpy(right.ptr<unsigned char>(0), output_r, nbytes);
    cv::resize(left, left, cv::Size(iw, ih), 0, 0, cv::INTER_LINEAR);
    cv::resize(right, right, cv::Size(iw, ih), 0, 0, cv::INTER_LINEAR);

    cv::imwrite((save0_dir + name).c_str(), left);
    cv::imwrite((save1_dir + name).c_str(), right);
    
    free(input_l);
    free(input_r);
    free(output_l);
    free(output_r);
  }

  UnInitialize();
}

void postProcess(const char *root_)
{
  const std::string root = root_;
  const std::string &cam0_dir = root + std::string("/device_1/");
  const std::string &visual_dir = root + std::string("/visual/");
  const std::string &config_file = root + std::string("/img.txt");

  std::ifstream fts;
  fts.open(config_file);
  if (!fts.is_open())
  {
    std::cerr << "img_pose path " << config_file.c_str() << "false\n";
    return ;
  }

  std::string name;
  std::set<uchar> inds;
  int id = 0, iw = 640, ih = 480;;
  cv::Mat visual_img = cv::Mat::zeros(ih, iw, CV_8UC3);
  while (!fts.eof())
  {
    fts >> name;

    cv::Mat left_in;
    left_in = cv::imread((cam0_dir + name).c_str(), 0);
    assert(left_in.type() == 0);

    if (!(id % 20))
    {
      for (int i = 0; i < ih; i++)
        for (int j = 0; j < iw; j++)
        {
          uchar ind = left_in.data[i * iw + j];
          inds.insert(ind);
          std::array<uchar, 3> color = getColor(ind);
          visual_img.data[i * iw * 3 + j * 3] = color[0];
          visual_img.data[i * iw * 3 + j * 3 + 1] = color[1];
          visual_img.data[i * iw * 3 + j * 3 + 2] = color[2];
        }
      cv::imwrite((visual_dir + name).c_str(), visual_img);

    }
    id++;
  }

  for (const uchar &it : inds)
    std::cout << it << " ";
  std::cout << std::endl;
}

int main(int argc, char **argv) {
  // segmentation(argv[1]);
  postProcess(argv[1]);
  return 0;
}
