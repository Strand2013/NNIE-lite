//
// Created by surui on 2020/7/1.
//


#include "GeneralSeg.h"


GeneralSeg::GeneralSeg(const std::string &modelPath)
{
    params.modelPath = modelPath;
    // You can reference your prototxt to fill these field.
    params.batchSize = 1;
    params.resizedHeight = 512;
    params.resizedWidth = 1024;
    params.inputC = 3;
    params.classNum = 33;

    if (!validateGparams(params))
    {
        perror("[ERROR] Check your gparams !\n\n");
    }
    net.load_model(params.modelPath.c_str());
}

void GeneralSeg::init(const std::string &modelPath)
{
    params.modelPath = modelPath;
    // You can reference your prototxt to fill these field.
    params.batchSize = 1;
    params.resizedHeight = 512;
    params.resizedWidth = 1024;
    params.inputC = 3;
    params.classNum = 33;

    if (!validateGparams(params))
    {
        perror("[ERROR] Check your gparams !\n\n");
    }
    net.load_model(params.modelPath.c_str());
}


bool GeneralSeg::validateGparams(nnie::gParams gparams)
{
    if (gparams.resizedHeight < 1 || gparams.resizedWidth < 1 || gparams.inputC < 1)
    {
        perror("[ERROR] You have to assign the resize info and channel !\n\n");
        return false;
    }
    if (gparams.batchSize < 1)
    {
        perror("[ERROR] You have to assign the batch size !\n\n");
        return false;
    }

    if (gparams.classNum < 1)
    {
        perror("[ERROR] You have to assign the class num !\n\n");
        return false;
    }
    if (gparams.modelPath.empty())
    {
        perror("[ERROR] You have to assign the engine path !\n\n");
        return false;
    }
    return true;
}

GeneralSeg::~GeneralSeg()
{
    net.clear();
}

void GeneralSeg::run(const std::string &imgPath, cv::Mat &clsIdxMask, cv::Mat &colorMask)
{
    cv::Mat im = params.inputC == 1 ? cv::imread(imgPath, 0): cv::imread(imgPath);
    run(im, clsIdxMask, colorMask);
}

void GeneralSeg::run(cv::Mat input_img, cv::Mat &clsIdxMask, cv::Mat &colorMask)
{
    if(input_img.rows != params.resizedHeight || input_img.cols != params.resizedWidth)
        cv::resize(input_img, input_img, cv::Size(params.resizedWidth, params.resizedHeight));

    nnie::Mat in;
    nnie::resize_bilinear(input_img, in, params.resizedWidth, params.resizedHeight, params.inputC);

    net.run(in.im);

    nnie::Mat logit;
    net.extract(0, logit);

#ifdef __DEBUG__
    printf("\nTensor h : %d\n", logit.height);
    printf("Tensor w : %d\n", logit.width);
    printf("Tensor c : %d\n", logit.channel);
    printf("Tensor n : %d\n", logit.n);
#endif

    clsIdxMask = cv::Mat::zeros(params.resizedHeight, params.resizedWidth, CV_8UC1);
    colorMask = cv::Mat::zeros(params.resizedHeight, params.resizedWidth, CV_8UC3);

    parseTensor(logit, clsIdxMask, colorMask);

#ifdef __DEBUG__
    cv::imwrite("color_mask.png", colorMask);
#endif

    free(in.im);

}

void GeneralSeg::parseTensor(nnie::Mat outTensor, cv::Mat clsIdxMask, cv::Mat &colorMask)
{

    float *res = outTensor.data;
    for (int i = 0; i < params.resizedHeight; ++i)
    {
        for (int j = 0; j < params.resizedWidth; ++j)
        {
            float max = -1;
            int maxIdx = 1;
            for (int c = 0; c < params.classNum; ++c)
            {
                float logit = res[j + (i * params.resizedWidth) + c * params.resizedHeight * params.resizedWidth];
//                printf("logit : %f\n", logit);
                if (logit > max)
                {
                    maxIdx = c;
                    max = logit;
                }
            }
//            printf("maxIdx : %f\n", maxIdx);
            clsIdxMask.at<uchar>(i, j) = maxIdx;
            colorMask.at<cv::Vec3b>(i, j) = colorMap[maxIdx];
        }
    }
}



// Example: General image segmeatation
// ========================= main =================================

int main()
{
    const std::string &pcModelPath = "./data/nnie_model/segmentation/ENet_cityscapes.wk";
    GeneralSeg enet;
    enet.init(pcModelPath);
    const std::string &pcSrcFile = "./data/nnie_image/cityscapes/lindau_000000_000019_leftImg8bit.png";
    cv::Mat clsMask;
    cv::Mat colorMask;
    enet.run(pcSrcFile, clsMask, colorMask);

}