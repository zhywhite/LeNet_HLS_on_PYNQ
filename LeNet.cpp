#include "LeNet.h"
#include "stdlib.h"
#include <time.h>
using namespace std;

//caculate exp use LUT
//maxpool LUT

FIX_ACC_MAX expf(FIX_ACC_MAX x)
{
	x = 1 + x / 1024;
	x *= x;
	x *= x;
	x *= x;
	x *= x;
	x *= x;
	x *= x;
	x *= x;
	x *= x;
	x *= x;
	x *= x;
	return x;
}

//
FIX_ACC Conv_5x5_BIG(FIX_ACC input[25], FIX_WT_BIG kernel[25])
{
	int x, y;
	FIX_ACC result = 0;
	for (y = 0; y < 25;)
	{
#pragma HLS PIPELINE
		for (x = 0; x < 5; x++)
		{
#pragma HLS UNROLL
			result += input[x + y] * kernel[x + y];
		}
		y = y + 5;
	}
	return result;
}

FIX_ACC Conv_5x5_SMALL(FIX_ACC input[25], FIX_WT_SMALL kernel[25])
{
	int x, y;
	FIX_ACC result = 0;
	for (y = 0; y < 25;)
	{
#pragma HLS PIPELINE
		for (x = 0; x < 5; x++)
		{
#pragma HLS UNROLL
			result += input[x + y] * kernel[x + y];
		}
		y = y + 5;
	}
	return result;
}

FIX_ACC relu_6(FIX_ACC x)
{
	FIX_RELU test=1,res;
	if(x > 0) {
		res = x >> 6;
		if ((test&res) == 1) res = res + 1;
		return res<<6;
	}
	else
	{
		return 0;
	}
}

FIX_ACC relu_4(FIX_ACC x)
{
	FIX_RELU test=1,res;
	if(x > 0) {
		res = x >> 4;
		if ((test&res) == 1) res = res + 1;
		return res<<4;
	}
	else
	{
		return 0;
	}
}

FIX_ACC relu_1(FIX_ACC x)
{
	FIX_RELU test=1,res;
	if(x > 0) {
		res = x >> 1;
		if ((test&res) == 1) res = res + 1;
		return res<<1;
	}
	else
	{
		return 0;
	}
}

FIX_ACC relu_0(FIX_ACC x)
{
	FIX_RELU test=1,res;
	if(x > 0) {
		res = x ;
		if ((test&res) == 1) res = res + 1;
		return res ;
	}
	else
	{
		return 0;
	}
}

//kernel weight 5x5x6 = 25x6 = 150
//template hanshumuban
void ConvLayer_1(FIX_FM input[1024], FIX_ACC *C1_value, FIX_WT_BIG *weights, FIX_BIAS *bias)
{
	int i_y, i_x, matrix_y, matrix_x;
	int k_num, mat_i = 0;
	FIX_WT_BIG matrix_2[25]; // weight kernel
    FIX_ACC matrix[25];  // photo
	int matrix_index;
    int input_value_index;
	int out_pic_index;
	int pic_value_index;
top_loop:
	for (int k_num = 0; k_num < 6; k_num += 1)
	{
		for (mat_i = 0; mat_i < 25; mat_i++)
		{
			matrix_2[mat_i] = weights[mat_i + k_num*25];
		}
	i_y_loop:
		for (i_y = 0; i_y < 28; i_y++)
		{
			for (i_x = 0; i_x < 28; i_x++)
			{
#pragma HLS PIPELINE
				pic_value_index = i_x + i_y * 32;
			matrix_loop:
				for (matrix_y = 0; matrix_y < 5; matrix_y++)
				{
				caculate:
					for (matrix_x = 0; matrix_x < 5; matrix_x++)
					{
						//						图片索引  0 ~ 24
					    matrix_index = matrix_x + matrix_y * 5;
						//						图片像素索引 0 ~ 1024,与matrix_x,matrix_y相关,x、y=32
						input_value_index = pic_value_index + matrix_x + matrix_y * 32;
						
						matrix[matrix_index] = input[input_value_index];
					}
				}
			    out_pic_index = i_x + i_y * 28 + k_num * 784;
				C1_value[out_pic_index] = relu_6(Conv_5x5_BIG(matrix, matrix_2) + bias[k_num]);
			}
		}
	}
}

// define max
FIX_ACC MaxPool_2x2(FIX_ACC input[4])
{
	FIX_ACC res,res1,res2;
    res1 = (input[0]>input[1])?input[0]:input[1];
	res2 = (input[2]>input[3])?input[2]:input[3];
	res = (res1>res2)?res1:res2;
	return res;
}

//relu
void MaxpoolLayer_2(FIX_ACC input[4704], FIX_ACC *A2_value)
{
	int k_num, i_y, i_x, matrix_x, matrix_y;
	int count = 0;
	int index_now;
	int input_index;
	FIX_ACC matrix[4];
	for (k_num = 0; k_num < 6; k_num++)
	{
		for (i_y = 0; i_y < 27; i_y += 2)
		{
			for (i_x = 0; i_x < 27; i_x += 2)
			{
				index_now = i_x + i_y * 28 + k_num * 784;
				for (matrix_y = 0; matrix_y < 2; matrix_y++)
				{
					for (matrix_x = 0; matrix_x < 2; matrix_x++)
					{
					    input_index = index_now + matrix_x + matrix_y * 28;
						matrix[matrix_x + matrix_y * 2] = input[input_index];
					}
				}
				A2_value[count] = MaxPool_2x2(matrix);
				count++;
			}
		}
	}
}

//kernel weight 5x5x6x16 = 25x6x16 =2400
// cut the data
// index
void ConvLayer_3(FIX_ACC input[1176], FIX_ACC *C3_value, FIX_WT_SMALL *weights, FIX_BIAS *bias)
{
	int k_num, nk_num, i_y, i_x, matrix_x, matrix_y;
	int mat_i;
    FIX_ACC res_total_6 = 0;
	FIX_WT_SMALL matrix_2[25]; FIX_ACC matrix[25];
	int index_now ; int weights_index;
	int matrix_index;  int input_value_index;
	for (nk_num = 0; nk_num < 16; nk_num++)
	{
		for (i_y = 0; i_y < 10; i_y++)
		{
			for (i_x = 0; i_x < 10; i_x++)
			{
#pragma HLS PIPELINE
				index_now = i_x + i_y * 10 + nk_num * 100;
				for (k_num = 0; k_num < 6; k_num++)
				{
					for (mat_i = 0; mat_i < 25; mat_i++)
					{
// memrcopy
					    weights_index = mat_i + k_num * 25 + nk_num * 150;
						matrix_2[mat_i] = weights[weights_index];
					}
// reorder
					for (matrix_y = 0; matrix_y < 5; matrix_y++)
					{
						for (matrix_x = 0; matrix_x < 5; matrix_x++)
						{
							matrix_index = matrix_x + matrix_y * 5;
							input_value_index = k_num * 196 + i_x + i_y * 14 + matrix_x + matrix_y * 14;
							matrix[matrix_index] = input[input_value_index];
						}
					}
					res_total_6 += Conv_5x5_SMALL(matrix, matrix_2);

				}
//				cout << res_total_6 << "\n" << endl;
                C3_value[index_now] = relu_4(res_total_6 + bias[6 + nk_num]);
				res_total_6 = 0;
			}
		}
	}
}

void MaxpoolLayer_4(FIX_ACC input[1600], FIX_ACC *A4_value)
{
	int k_num, i_y, i_x, matrix_x, matrix_y;
	int count = 0;
	FIX_ACC matrix[4];
	int index_now;  int input_index;
	for (k_num = 0; k_num < 16; k_num++)
	{
		for (i_y = 0; i_y < 10; i_y += 2)
		{
			for (i_x = 0; i_x < 10; i_x += 2)
			{
				index_now = i_x + i_y * 10 + k_num * 100;
				for (matrix_y = 0; matrix_y < 2; matrix_y++)
				{
					for (matrix_x = 0; matrix_x < 2; matrix_x++)
					{
						input_index = index_now + matrix_x + matrix_y * 10;
						matrix[matrix_x + matrix_y * 2] = input[input_index];
					}
				}
				A4_value[count] = MaxPool_2x2(matrix);
				count++;
			}
		}
	}
}

//kernel 120*400 = 48000
// add
void FullyConnLayer_5(FIX_ACC input[400], FIX_ACC *F5_value, FIX_WT_SMALL *weights, FIX_BIAS *bias)
{
	int i_y, i_x;
	FIX_ACC_MAX result = 0;
	int index;
	FIX_ACC res1;
	for (i_y = 0; i_y < 120; i_y++)
	{
		for (i_x = 0; i_x < 400; i_x++)
		{
#pragma HLS UNROLL factor=50
			index = i_x + i_y * 400;
			result += input[i_x] * weights[index + 2400];
		}
		// 6 + 16 = 22
		F5_value[i_y] = relu_1(result + bias[i_y + 22]);
		result = 0;
	}
}

//kernel 84x120 = 10080

void FullyConnLayer_6(FIX_ACC input[120], FIX_ACC *F6_value, FIX_WT_SMALL *weights, FIX_BIAS *bias)
{
	int i_y, i_x;
	FIX_ACC_MAX result = 0;
	FIX_ACC res1=0 ;
	int index;
	for (i_y = 0; i_y < 84; i_y++)
	{
		for (i_x = 0; i_x < 120; i_x++)
		{
#pragma HLS UNROLL factor=30
			index = i_x + i_y * 120;
			result += input[i_x] * weights[index + 50400];
		}
		// 22 + 120 = 142
		F6_value[i_y] = relu_0(result + bias[142 + i_y]);
		result = 0;
	}
}

//kernel 10x84 = 840
void FullyConnLayer_7(FIX_ACC input[84], FIX_ACC_MAX *F7_value, FIX_WT_SMALL *weights, FIX_BIAS *bias)
{
	int i_y, i_x;
    FIX_ACC_MAX res = 0;
	int index;
	for (i_y = 0; i_y < 10; i_y++)
	{
		for (i_x = 0; i_x < 84; i_x++)
		{
#pragma HLS UNROLL factor=21
			index = i_x + i_y * 84;
			res += input[i_x] * weights[index + 60480];
		}
		// 142 + 84 = 226
		F7_value[i_y] = res + bias[226 + i_y];
		res =0 ;
	}
}

void Softmax_1_8(FIX_RESULT r[1],FIX_ACC_MAX input[10])
{
	int index;
	FIX_ACC_MAX sum = 0;
	FIX_ACC_MAX temp;
	FIX_ACC_MAX probability[10];
	FIX_ACC_MAX res[10];
	for (index = 0; index < 10; index++)
	{
		temp = input[index] >> 4;
		probability[index] = expf(temp);
		sum += probability[index];
	}
	int max_index = 0;
	FIX_ACC_MAX res1 ;
	FIX_ACC_MAX res2 ;
	for (index = 0; index < 10; index++)
	{
		res[index] = probability[index] / sum;
		res1 = res[index];
	    res2 = res[max_index];
		if (res1 > res2)
		{
			max_index = index;
		}
	}
	r[0] = max_index;
}

void LeNet(FIX_WT_BIG weight_big[150],FIX_WT_SMALL weight_small[61320], FIX_FM photo[1024], FIX_RESULT r[1], FIX_BIAS bias[236])
{
#pragma HLS INTERFACE m_axi depth=150 port=weight_big bundle=DATA
#pragma HLS INTERFACE m_axi depth=61320 port=weight_small bundle=DATA
#pragma HLS INTERFACE m_axi depth=1024 port=photo bundle=DATA
#pragma HLS INTERFACE m_axi depth=236 port=bias bundle=DATA
#pragma HLS INTERFACE m_axi depth=1 port=r bundle=DATA


#pragma HLS INTERFACE s_axilite register port=bias bundle=CTRL
#pragma HLS INTERFACE s_axilite register port=weight_small bundle=CTRL
#pragma HLS INTERFACE s_axilite register port=weight_big bundle=CTRL
#pragma HLS INTERFACE s_axilite register port=photo bundle=CTRL
#pragma HLS INTERFACE s_axilite register port=r bundle=CTRL


#pragma HLS INTERFACE s_axilite register port=return bundle=CTRL
	//layer1 weights  5x5x6 = 25x6 = 150
	//layer3 weights  5x5x6x16 = 25x6x16 =2400
	//layer5 weights 400x120 = 48000
	//layer6 weights 84x120 = 10080
	//layer7 weights 10x84 = 840

	//The output of each layer
	FIX_ACC C1_value[4704];
	FIX_ACC A2_value[1176];
	FIX_ACC C3_value[1600];
	FIX_ACC A4_value[400];
	FIX_ACC F5_value[120];
	FIX_ACC F6_value[84];
	FIX_ACC_MAX F7_value[10];
	//calulation of each layer
	ConvLayer_1(photo, C1_value, weight_big, bias);
	MaxpoolLayer_2(C1_value, A2_value);
	ConvLayer_3(A2_value, C3_value, weight_small, bias);
	MaxpoolLayer_4(C3_value, A4_value);
	FullyConnLayer_5(A4_value, F5_value, weight_small, bias);
	FullyConnLayer_6(F5_value, F6_value, weight_small, bias);
	FullyConnLayer_7(F6_value, F7_value, weight_small, bias);
	Softmax_1_8(r,F7_value);
}
