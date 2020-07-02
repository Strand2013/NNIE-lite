//
// Created by surui on 2020/7/1.
//

#ifndef NNIE_LITE_GENERALCLS_H
#define NNIE_LITE_GENERALCLS_H

#include <iostream>
#include <Net.hpp>
#include <unordered_map>
#include <opencv2/opencv.hpp>

class GeneralCls
{

public:

    GeneralCls()
    {};

    GeneralCls(const std::string &modelPath);

    ~GeneralCls();

    void init(const std::string &modelPath);

    // input absolute path of image
    void run(const std::string &imgPath, nnie::clsRes &clsInfo);

    // input cv::Mat
    void run(cv::Mat input_img, nnie::clsRes &clsInfo);


private:

    nnie::gParams params;

    nnie::Net net;

private:
    /**
        * Check the valid of parameters.
        * @param gparams
        * @return
        */
    bool validateGparams(nnie::gParams gparams);


    /**
     * The function is used to parse output tensor from NNIE.
     * clsIdxMask is a one channel matrix which each pixel represent a classification.
     * colorMap is a three channel matrix, it's a rgb picture for visualization.
     *
     * @param outTensor
     * @param clsIdxMask
     * @param colorMap
     */
    void parseTensor(nnie::Mat outTensor, cv::Mat clsIdxMask, cv::Mat &colorMask);
};


#endif //NNIE_LITE_GENERALCLS_H
