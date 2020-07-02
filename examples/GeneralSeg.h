//
// Created by surui on 2020/7/1.
//

#ifndef NNIE_LITE_GENERALSEG_H
#define NNIE_LITE_GENERALSEG_H

#include <iostream>
#include <Net.hpp>
#include <unordered_map>
#include <opencv2/opencv.hpp>

class GeneralSeg
{

public:

    GeneralSeg(){};

    GeneralSeg(const std::string &modelPath);

    ~GeneralSeg();

    void init(const std::string &modelPath);

    // input absolute path of image
    void run(const std::string &imgPath, cv::Mat &clsIdxMask, cv::Mat &colorMask);

    // input cv::Mat
    void run(cv::Mat input_img, cv::Mat &clsIdxMask, cv::Mat &colorMask);


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

private:
    nnie::gParams params;

    nnie::Net net;

    std::unordered_map<int, cv::Vec3b> colorMap = {
            {0, {0, 0, 0}},
            {1, {219, 0, 255}},
            {2, {0, 255, 132}},
            {3, {255, 0, 15}},
            {4, {255, 214, 0}},
            {5, {255, 91, 0}},
            {6, {255, 76, 0}},
            {7, {0, 255, 239}},
            {8, {127, 255, 0}},
            {9, {0, 255, 117}},
            {10, {234, 0, 255}},
            {11, {255, 137, 0}},
            {12, {0, 117, 255}},
            {13, {35, 0, 255}},
            {14, {0, 255, 71}},
            {15, {5, 0, 255}},
            {16, {255, 198, 0}},
            {17, {0, 147, 255}},
            {18, {255, 0, 198}},
            {19, {0, 255, 147}},
            {20, {203, 255, 0}},
            {21, {255, 244, 0}},
            {22, {255, 0, 229}},
            {23, {255, 61, 0}},
            {24, {219, 255, 0}},
            {25, {234, 255, 0}},
            {26, {204, 0, 255}},
            {27, {255, 15, 0}},
            {28, {81, 255, 0}},
            {29, {142, 255, 0}},
            {30, {112, 0, 255}},
            {31, {142, 0, 255}},
            {32, {255, 0, 0}},
            {33, {255, 107, 0}},
            {34, {255, 30, 0}},
            {35, {249, 0, 255}},
            {36, {0, 178, 255}},
            {37, {0, 255, 178}},
            {38, {51, 255, 0}},
            {39, {0, 255, 56}},
            {40, {173, 255, 0}},
            {41, {0, 25, 255}},
            {42, {81, 0, 255}},
            {43, {255, 0, 76}},
            {44, {0, 255, 25}},
            {45, {0, 10, 255}},
            {46, {158, 0, 255}},
            {47, {255, 153, 0}},
            {48, {35, 255, 0}},
            {49, {0, 86, 255}},
            {50, {255, 0, 30}},
            {51, {255, 0, 107}},
            {52, {249, 255, 0}},
            {53, {0, 163, 255}},
            {54, {0, 255, 209}},
            {55, {255, 0, 45}},
            {56, {255, 0, 152}},
            {57, {255, 122, 0}},
            {58, {20, 255, 0}},
            {59, {0, 255, 40}},
            {60, {255, 183, 0}},
            {61, {255, 45, 0}},
            {62, {96, 255, 0}},
            {63, {66, 255, 0}},
            {64, {0, 255, 86}},
            {65, {0, 224, 255}},
            {66, {255, 168, 0}},
            {67, {0, 255, 102}},
            {68, {0, 132, 255}},
            {69, {255, 229, 0}},
            {70, {0, 255, 193}},
            {71, {188, 0, 255}},
            {72, {0, 193, 255}},
            {73, {255, 0, 61}},
            {74, {0, 255, 224}},
            {75, {0, 255, 163}},
            {76, {0, 255, 10}},
            {77, {255, 0, 137}},
            {78, {158, 255, 0}},
            {79, {255, 0, 91}},
            {80, {0, 56, 255}},
            {81, {112, 255, 0}},
            {82, {96, 0, 255}},
            {83, {0, 209, 255}},
            {84, {0, 40, 255}},
            {85, {173, 0, 255}},
            {86, {0, 71, 255}},
            {87, {255, 0, 168}},
            {88, {255, 0, 214}},
            {89, {255, 0, 183}},
            {90, {127, 0, 255}},
            {91, {5, 255, 0}},
            {92, {0, 255, 255}},
            {93, {188, 255, 0}},
            {94, {255, 0, 244}},
            {95, {20, 0, 255}},
            {96, {0, 239, 255}},
            {97, {66, 0, 255}},
            {98, {255, 0, 122}},
            {99, {0, 102, 255}},
            {100, {50, 0, 255}},
    };
};


#endif //NNIE_LITE_GENERALSEG_H
