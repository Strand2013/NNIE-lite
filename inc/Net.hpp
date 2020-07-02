//
// Created by surui on 2020/6/19.
//

#ifndef NNIE_LITE_NET_HPP
#define NNIE_LITE_NET_HPP

#include <stdio.h>
#include <nnie_core.h>
#include "sample_comm_nnie.h"
#include <opencv2/opencv.hpp>

#define MAX_OUTPUT_NUM 5


namespace nnie
{
    typedef struct Params
    {
        // required
        std::string modelPath{};
        int batchSize{0};
        int resizedHeight{0};
        int resizedWidth{0};
        int inputC{0};
        int classNum{0};
        //optional
        std::vector<int> outputNodeName{0};

    } gParams;

    // analogous to ncnn
    typedef Tensor_S Mat;

    // Used for saving classification's result
    typedef struct ClsRes
    {
        int index;
        std::vector<float> prob;
    } clsRes;

    typedef struct ObjectDetection_S
    {
        int cls_id;
        float confidence;
        cv::Rect2d box;
    } objInfo;

    void resize_bilinear(const cv::Mat &src, Mat &dst, int w, int h, int c);

    class Net
    {

    public:
        Net();

        ~Net();

    public:
        // analogous to ncnn
        int load_model(const char *model_path);

        int extract(int blob_name, Mat &feat);

        void clear();

    public:
        void run(const char *file_path);

        void run(const unsigned char *data);

        void finish();

        Mat getOutputTensor(int index);


    protected:

        Mat output_tensors_[MAX_OUTPUT_NUM];

        SAMPLE_SVP_NNIE_MODEL_S s_stModel_;
        SAMPLE_SVP_NNIE_PARAM_S s_stNnieParam_;
        SAMPLE_SVP_NNIE_CFG_S stNnieCfg_;


    };
}
#endif //NNIE_LITE_NET_HPP
