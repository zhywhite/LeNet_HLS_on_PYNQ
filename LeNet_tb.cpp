#include "LeNet.h"
#include "stdlib.h"
using namespace std;

int main(){
	FIX_WT_SMALL weight_small[61320];
	FIX_WT_BIG weight_big[150];
	FIX_FM photo[1024];
	FIX_BIAS bias[236];
	FIX_RESULT r[1];
	for(long i_2 = 0; i_2<150; i_2++){
	//	cout <<"weight" <<weights[i_2] << "\n" << endl;
	    weight_big[i_2] = 4;
	}
	for(long i_2 = 0; i_2<61320; i_2++){
	//	cout <<"weight" <<weights[i_2] << "\n" << endl;
	    weight_small[i_2] = 0.01;
	}
	for(int i_1 = 0; i_1<1024; i_1++){
		photo[i_1] = 0.8;
	//    cout <<"photo"  <<photo[i_1] << "\n" << endl;
	}
	for(int i_1 = 0; i_1<236; i_1++){
		bias[i_1] = 1;
	//    cout << "bias" <<bias[i_1] << "\n" << endl;
	}
	LeNet(weight_big,weight_small,photo,r,bias);
	//cout << weights[0] << endl;
	cout<<"res:"<<r[0]<<"\n" << endl;
	return 0;
}
