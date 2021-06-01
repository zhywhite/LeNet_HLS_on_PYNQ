//#include "stdio.h"
//#include <hls_math.h>

#include "iostream"
#include "ap_int.h"
#include "ap_fixed.h"

// define the weight and feature map size
#define FM_SIZE 8
#define WEIGHT_SIZE 8
#define BIAS_SIZE 16
#define FM_ACC_SMALL_SIZE 16
#define FM_ACC_BIG_SIZE 16
#define FM_ACC_MAX_SIZE 32

// define data type
typedef ap_fixed<FM_SIZE,1,AP_RND,AP_SAT> FIX_FM;
typedef ap_fixed<WEIGHT_SIZE,5,AP_RND,AP_SAT> FIX_WT_BIG;
typedef ap_fixed<WEIGHT_SIZE,1,AP_RND,AP_SAT> FIX_WT_SMALL;
typedef ap_fixed<BIAS_SIZE,5,AP_RND,AP_SAT> FIX_BIAS;
typedef ap_fixed<FM_ACC_SMALL_SIZE,9,AP_RND,AP_SAT> FIX_ACC;
typedef ap_fixed<FM_ACC_MAX_SIZE,16,AP_RND,AP_SAT> FIX_ACC_MAX;
typedef ap_int<16> FIX_RELU;
typedef ap_int<8> FIX_RESULT;


void LeNet(FIX_WT_BIG weight_big[150],FIX_WT_SMALL weight_small[61320], FIX_FM photo[1024], FIX_RESULT r[1], FIX_BIAS bias[236]);
