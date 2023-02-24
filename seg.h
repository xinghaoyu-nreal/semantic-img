#pragma once

extern "C" {

bool InitPlugin();
bool UnInitialize();
bool InferenceSeg(const unsigned char *input_image, int width, int height,
                  unsigned char *output_image, int &out_width, int &out_height);
}
