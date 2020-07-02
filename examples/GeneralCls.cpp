//
// Created by surui on 2020/7/1.
//

#include "GeneralCls.h"


GeneralCls::GeneralCls(const std::string &modelPath)
{
    params.modelPath = modelPath;
    // You can reference your prototxt to fill these field.
    params.batchSize = 1;
    params.resizedHeight = 28;
    params.resizedWidth = 28;
    params.inputC = 1;
    params.classNum = 10;

    if (!validateGparams(params))
    {
        perror("[ERROR] Check your gparams !\n\n");
    }
    net.load_model(params.modelPath.c_str());
}


void GeneralCls::init(const std::string &modelPath)
{
    params.modelPath = modelPath;
    // You can reference your prototxt to fill these field.
    params.batchSize = 1;
    params.resizedHeight = 28;
    params.resizedWidth = 28;
    params.inputC = 1;
    params.classNum = 10;

    if (!validateGparams(params))
    {
        perror("[ERROR] Check your gparams !\n\n");
    }
    net.load_model(params.modelPath.c_str());
}


bool GeneralCls::validateGparams(nnie::gParams gparams)
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

GeneralCls::~GeneralCls()
{
    net.clear();
}

void GeneralCls::run(const std::string &imgPath, nnie::clsRes &clsInfo)
{
    cv::Mat im = params.inputC == 1 ? cv::imread(imgPath, 0): cv::imread(imgPath);
    run(im, clsInfo);
}

void GeneralCls::run(cv::Mat input_img, nnie::clsRes &clsInfo)
{
    if (input_img.rows != params.resizedHeight || input_img.cols != params.resizedWidth)
        cv::resize(input_img, input_img, cv::Size(params.resizedWidth, params.resizedHeight));

    nnie::Mat in;
    nnie::resize_bilinear(input_img, in, params.resizedWidth, params.resizedHeight, params.inputC);

    net.run(in.im);

    nnie::Mat logit;
    net.extract(0, logit);

#ifdef __DEBUG__
    printf("\n\nTensor h : %d\n", logit.height);
    printf("Tensor w : %d\n", logit.width);
    printf("Tensor c : %d\n", logit.channel);
    printf("Tensor n : %d\n", logit.n);
#endif

    float *prob = logit.data;
    int maxIdx = 0;
    float maxProb = -1;
    for (int i = 0; i < logit.width; ++i)
    {
#ifdef __DEBUG__
        printf(" %f ", *prob);
#endif
        if (*prob > maxProb)
        {
            clsInfo.prob.push_back(*prob);
            maxProb = *prob;
            maxIdx = i;
        }
        prob++;
    }
    clsInfo.index = maxIdx;
    free(in.im);
}


// Example: General image classification
// ========================= main =================================

int main()
{
    const std::string &pcModelPath = "./data/nnie_model/mnist/inst_mnist_cycle.wk";
    GeneralCls mnist;
    mnist.init(pcModelPath);
    const std::string &pcSrcFile = "./data/nnie_image/mnist/8_102636.jpg";
    nnie::ClsRes cls_res;
    mnist.run(pcSrcFile, cls_res);

    printf("\n class : %d \n", cls_res.index);

}


