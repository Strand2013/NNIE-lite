#include "stdio.h"
#include "Net.hpp"
#include "nnie_core.h"


namespace nnie
{
    void resize_bilinear(const cv::Mat &src, Mat &dst, int w, int h, int c)
    {
        cv::Mat im = src.clone();
        if (src.rows != h || src.cols != w)
        {
            cv::resize(src, im, cv::Size(w, h));
            printf("resize done.");
        }

        dst.height = h;
        dst.width = w;
        dst.channel = c;

        unsigned char *data = (unsigned char *) malloc(sizeof(unsigned char) * h * w * c);

        int step = im.step;
        unsigned char *data1 = (unsigned char *) im.data;
        int count = 0;
        for (int k = 0; k < c; k++)
        {
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    data[count++] = data1[i * step + j * c + k];
                }
            }
        }

        dst.im = data;
    }

    Net::Net()
    {}

    Net::~Net()
    {
        NNIE_Param_Deinit(&s_stNnieParam_, &s_stModel_);
    }

    int Net::load_model(const char *model_path)
    {
        _load_model(model_path, &s_stModel_);
        nnie_param_init(&s_stModel_, &stNnieCfg_, &s_stNnieParam_);
    }

    void Net::run(const char *file_path)
    {

        int file_length = 0;
        FILE *fp = fopen(file_path, "rb");
        if (fp == NULL)
        {
            printf("open %s failed\n", file_path);
            return;
        }

        fseek(fp, 0L, SEEK_END);
        file_length = ftell(fp);
        fseek(fp, 0L, SEEK_SET);

        unsigned char *data = (unsigned char *) malloc(sizeof(unsigned char) * file_length);

        fread(data, file_length, 1, fp);

        fclose(fp);

        NNIE_Forward_From_Data(data, &s_stModel_, &s_stNnieParam_, output_tensors_);

        free(data);
    }

    void Net::run(const unsigned char *data)
    {
        NNIE_Forward_From_Data(data, &s_stModel_, &s_stNnieParam_, output_tensors_);
    }

    void Net::finish()
    {
        NNIE_Param_Deinit(&s_stNnieParam_, &s_stModel_);
    }

    // analogous to ncnn
    void Net::clear()
    {
        NNIE_Param_Deinit(&s_stNnieParam_, &s_stModel_);
    }


    int Net::extract(int blob_name, Tensor &feat)
    {
        feat = output_tensors_[blob_name];
        return 0;
    }

    Mat Net::getOutputTensor(int index)
    {
        return output_tensors_[index];
    }


}

