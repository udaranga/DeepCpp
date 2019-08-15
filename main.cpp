//
// Created by Udaranga Wickramasinghe on 15.08.19.
//


#include <stdio.h>
#include <iostream>
#include <math.h>
#include <cmath>
#include <cassert>
using namespace std;

float sigmoid(float x)
{
    float exp_value;
    float return_value;

    /*** Exponential calculation ***/
    exp_value = (float)exp((double)-x);

    /*** Final sigmoid value ***/
    return_value = 1 / (1 + exp_value);

    return return_value;
}

float ReLU(float x)
{
    return (x > 0 ? x : 0);
}

float same(float x)
{
    return x;
}



/** Data structure used to store data for convolutions. **/
class Matrix_info
{
private:

    int _depth, _height, _width, _channels;


public:
    //float * _data;
    void set(float* data, float vlaue, int c, int d, int h, int w); // c is channel count, d,h,w are spatial dims
    void set(float* data, float value, int c);
    float get(float* data, int c, int d, int h, int w);
    float get(float* data, int c);
    int depth();
    int height();
    int width();
    int channels();

    void set_dims(int channels, int depth, int height, int width);
    void set_dims(int channels);

    int size();
    Matrix_info(int channels, int depth, int height, int width);
    Matrix_info(int channels);
    Matrix_info();

};

/** Default constructor **/
Matrix_info::Matrix_info() {}



/** Empty matrix - used to initilize results matrix **/
Matrix_info::Matrix_info(int channels, int depth, int height, int width)
{
    this->_channels = channels;
    this->_depth = depth;
    this->_height = height;
    this->_width = width;

}

Matrix_info::Matrix_info(int channels)
{
    this->_height = 1;
    this->_width = 1;
    this->_depth = 1;
    this->_channels = channels;
}



/** setters **/
void Matrix_info::set(float *data, float value, int c, int d, int h, int w) {
    if (c < channels() && d < depth() && h < height() && w < width() && c >= 0 && d >= 0 && h >= 0 && w >= 0)
        data[c * this->_depth * this->_height * this->_width + d * this->_height * this->_width + h * this->_width + w] = value;
    else
        assert (0);
}

/** setters **/
void Matrix_info::set(float* data, float value, int c) {
    if (c < this->channels() && c >= 0)
        data[c] = value;
    else
        assert (0);
}

/** getters **/
float Matrix_info::get(float* data, int c, int d, int h, int w) {
    if (c < channels() && d < depth() && h < height() && w < width() && c >= 0 && d >= 0 && h >= 0 && w >= 0)
        return data[c * this->_depth * this->_height * this->_width + d * this->_height * this->_width + h * this->_width + w];
    else
        return nanf("1");
}

/** getters **/
float Matrix_info::get(float* data, int c) {
    if (c < this->channels() && c >= 0)
        return data[c];
    else
        return nanf("2");
}

void Matrix_info::set_dims(int channels, int depth, int height, int width)
{
    this->_channels = channels;
    this->_depth = depth;
    this->_height = height;
    this->_width = width;
}


void Matrix_info::set_dims(int c)
{
    this->_channels = c;
    this->_depth = 1;
    this->_height = 1;
    this->_width = 1;
}


int Matrix_info::size()
{
    return this->_channels * this->_depth * this->_height * this-> _width;
}

int Matrix_info::channels() {return this->_channels;}
int Matrix_info::depth() {return this->_depth;}
int Matrix_info::height() {return this->_height;}
int Matrix_info::width() {return this->_width;}



/*** Convolution layer ***/
class Conv3d {
private:
    float* W; /** Kernel  dimensions: Output_channel_count x Input_channel_count x Kenrel_Height x Kernel_width   **/
    float* b; /** bias    dimensions: Output_channel_count                                                        **/
    float (*activation)(float);

public:
    int C_in, C_out;
    int K_depth, K_height, K_width;
    Conv3d();
    ~Conv3d();
    void initialize(int C_in, int C_out, int K_depth, int K_height, int K_width, float (*activation)(float));
    void set(float* W, float* b);
    float* forward(float* input_data, Matrix_info& x);
};


Conv3d::Conv3d() {
}

Conv3d::~Conv3d() {
    delete [] W;
    delete [] b;
}

void Conv3d::initialize(int C_in, int C_out, int K_depth, int K_height, int K_width, float (*activation)(float)) {
    this->C_in = C_in;
    this->C_out = C_out;
    this->K_depth = K_depth;
    this->K_height = K_height;
    this->K_width = K_width;
    this->activation = activation;

    W = new float[C_in * C_out * K_depth * K_height * K_width];
    b = new float[C_out];
}

void Conv3d::set(float* W, float* b) {
    for (int i = 0; i < C_in * C_out * K_depth * K_height * K_width; i++)
        this->W[i] = W[i];

    for (int i = 0; i < C_out; i++)
        this->b[i] = b[i];

    delete [] W;
    delete [] b;
}

float* Conv3d::forward(float* input_data, Matrix_info& info) {
    int input_channels = info.channels();
    int input_depth = info.depth();
    int input_height = info.height();
    int input_width = info.width();



    if (C_in != input_channels) // input matrix number of channels MUST EQUAL kernel depth
        cout << "Error!-";

    int output_depth = input_depth - 2 * (K_depth / 2); // intiger division = floor(K_height/2)
    int output_height = input_height - 2 * (K_height / 2); // intiger division = floor(K_height/2)
    int output_width = input_width - 2 * (K_width / 2); // intiger division = floor(K_height/2)

    info.set_dims(C_out, output_depth, output_height, output_width);
    float* results_data = new float[C_out * output_width * output_height  * output_width];

    for (int i=0; i< C_out * output_height  * output_width;i++)
        results_data[i] = 0;

    for (int c_out = 0; c_out < C_out; c_out++ )
        for (int k_out = 0; k_out < output_depth; k_out++)
            for (int i_out = 0; i_out < output_height; i_out++)
                for (int j_out = 0; j_out < output_width; j_out++)
                {
                    float value = 0;
                    for (int c_in = 0; c_in < C_in; c_in++)
                        for (int kp = -K_depth / 2; kp < K_depth / 2 + 1; kp++)
                            for (int ip = -K_height / 2; ip < K_height / 2 + 1; ip++)
                                for (int jp = -K_width / 2; jp < K_width / 2 + 1; jp++)
                                    value += input_data[c_in * input_depth * input_height * input_width + (k_out + K_depth / 2 + kp) * input_height * input_width + (i_out + K_height / 2 + ip) * input_width + j_out + K_width / 2 + jp] *
                                             this->W[c_out * C_in * K_depth * K_height * K_width + c_in * K_depth * K_height * K_width +
                                                     (K_depth / 2 + kp) * K_width * K_height + (K_height / 2 + ip) * K_width + (K_width / 2 + jp)];

                    results_data[c_out * info.depth() * info.height() * info.width() + k_out * info.height() * info.width() + i_out * info.width() + j_out] = activation(value + this->b[c_out]);
                }
    delete [] input_data;
    return results_data;
}




/*** FC layer ***/
// TODO: Make it compatible with Matrix_info instead of float *
class Linear {
private:
    float* W; /** L_out _data L_in dim **/
    float* b; /** L_out _data 1 dim **/
    int L_in, L_out;
    float* multiply(float* data, float *W);
    void add(float* Wx, float *b);
    void activation(float* Wxb);
    float (*activation_function)(float);

public:
    Linear();
    ~Linear();
    void initialize(int L_in, int L_out, float (*activation)(float));
    void set(float* W, float* b);
    float* forward(float* data, Matrix_info& x);
};

Linear::Linear() {
    W = NULL;
    b = NULL;
}

Linear::~Linear() {
    delete [] W;
    delete [] b;
}
void Linear::initialize(int L_out, int L_in, float (*activation_function)(float)) {
    W = NULL;
    b = NULL;

    this->L_in = L_in;
    this->L_out = L_out;
    this->activation_function = activation_function;

    W = new float[L_in * L_out];
    b = new float[L_out];
}

void Linear::set(float* W, float* b) {


    for (int i = 0; i < L_in * L_out; i++)
        this->W[i] = W[i];

    for (int i = 0; i < L_out; i++)
        this->b[i] = b[i];

    delete [] W;
    delete [] b;
}

float* Linear::forward(float* input, Matrix_info& info) {

    info.set_dims(L_out);
    float* output = multiply(input, W);
    add(output, b);
    activation(output);

    return output;
}

float* Linear::multiply(float* data, float *W) {
    float* results = new float[L_out];

    for (int i=0; i< L_out;i++)
        results[i] = 0;

    /** Multiply **/
    for (int i = 0; i < L_out; i++)
        for (int j = 0; j < L_in; j++)
            results[i] = results[i] + W[i*L_in + j] * data[j];

    delete [] data;
    return results;
}

void Linear::add(float* Wx, float *b) {
    /** Add **/
    for (int i = 0; i < L_out; i++)
        Wx[i] = Wx[i] + b[i];
}

void Linear::activation(float* Wxb) {
    for (int i = 0; i < L_out; i++)
        Wxb[i] = activation_function(Wxb[i]);
}






class Network
{
private:
    static const int Conv_count = 3;
    int Conv_kerneldim[Conv_count][5] = { {1, 4, 1, 5, 1}, {4, 4, 1, 1, 3}, {4, 4, 1, 1, 3} }; // {{in_channel_count, out_channel_count, kernel_depth, kernel_height, kernel_width},...}

    static const int fc_count = 3; // length of archi - 1
    int fc_archi[fc_count + 1] = { 8, 6, 4, 3 }; // 50 is the input number of features, 10 is output number of features

//	Linear fc_layers[fc_count];
//	Conv2d conv_layers[Conv_count];
    Linear* fc_layers;
    Conv3d* conv_layers;

public:
    Network();
    float* forward(float* data, Matrix_info info);
    void set(float *E);
    int param_count();
};

Network::Network() {

    /*** Check validity of conv kernel sizes ***/
    for (int i = 0; i < Conv_count-1; i++)
        if (Conv_kerneldim[i][1] != Conv_kerneldim[i+1][0])
            assert(0);


    conv_layers = new Conv3d[Conv_count];
    for (int i = 0; i < Conv_count; i++)
        conv_layers[i].initialize(Conv_kerneldim[i][0], Conv_kerneldim[i][1], Conv_kerneldim[i][2], Conv_kerneldim[i][3], Conv_kerneldim[i][4], ReLU);

    fc_layers = new Linear[fc_count];
    for (int i = 0; i < fc_count; i++)
        fc_layers[i].initialize(fc_archi[i + 1], fc_archi[i],  i == fc_count-1 ? sigmoid :ReLU); //
}


float* Network::forward(float* data, Matrix_info info)
{

    for (int i = 0; i < Conv_count; i++)
        data = conv_layers[i].forward(data, info);

    /*** Number of features after last conv layer MUST EQUAL input feature count of the FC layer ***/
    if (info.size() != fc_archi[0])
        assert (0);

    for (int i = 0; i < fc_count; i++)
        data = fc_layers[i].forward(data, info);

    return data;
}

void Network::set(float* E)
{
    int used = 0;

    for (int i = 0; i < Conv_count; i ++)
    {
        float * W_i;
        int W_i_param_count = conv_layers[i].C_out * conv_layers[i].C_in * conv_layers[i].K_depth * conv_layers[i].K_height * conv_layers[i].K_width ;
        W_i = new float[W_i_param_count];
        for (int j = 0; j < W_i_param_count; j++)
            W_i[j] = E[j + used];

        used += W_i_param_count;

        float * b_i;
        int b_i_param_count = conv_layers[i].C_out;
        b_i = new float[b_i_param_count];
        for (int j = 0; j < b_i_param_count; j++)
            b_i[j] = E[j + used];

        used += b_i_param_count;
        conv_layers[i].set(W_i, b_i);

        // TODO: delete the arrays (gc)
    }
    for (int i = 0; i < fc_count; i++)
    {
        float * W_i;
        W_i = new float[fc_archi[i + 1] * fc_archi[i]];
        for (int j = 0; j < fc_archi[i + 1] * fc_archi[i]; j++)
            W_i[j] = E[j + used];
        used += fc_archi[i + 1] * fc_archi[i];

        float * b_i;
        b_i = new float[fc_archi[i + 1]];
        for (int j = 0; j < fc_archi[i + 1]; j++)
            b_i[j] = E[j + used];
        used += fc_archi[i + 1];

        fc_layers[i].set(W_i, b_i);

        // TODO: delete the arrays (gc)
    }
}


int main()
{

    Network net;
/*	float x[30] = { 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
		3.0, 3.0, 4.0, 2.0, 2.0, 2.0,
		5.0, 6.0, 7.0, 2.0, 2.0, 2.0,
		8.0, 9.0, 10.431, 2.0, 2.0, 2.0,
		8.0, 9.8439, 10.0, 2.0, 2.0, 2.351};*/



//	float E[217] = {0.256,0.131,0.536,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
//	float E[97] = {0.0505617071429,0.499951333238,-0.995908931107,0.693598508291,-0.418301520027,-1.58457723511,-0.647706767122,0.598575173967,0.33225003261,-1.14747663295,0.618669689027,-0.0879869283403,0.425072396487,0.332253145372,-1.15681626092,0.350997153083,-0.606887283097,1.54697932902,0.723341608748,0.0461355672302,-0.982991653419,0.0544327388652,0.159892935073,-1.20894815913,2.22336021691,0.394295214714,1.69235771524,-1.11281215385,1.6357475418,-1.36096559184,-0.65122583333,0.542451308403,0.0480062471907,-2.35807363301,-1.10558404382,0.83783635392,2.08787086828,0.914840957801,-0.276203354265,0.796511898725,-1.14379857185,0.50991978297,-1.34746029509,-0.00936010062991,-0.130704638588,0.802086613629,-0.302963967128,1.20200258981,-0.196745278478,0.836528702131,0.786602282754,-1.8408758668,0.0375474865106,0.0359280514159,-0.778739924557,0.179410714357,-1.4555343272,0.55618522295,0.509778854538,0.300445542948,2.47658416145,0.352343396548,0.0674710005507,-0.732264699769,0.29714121028,-0.961776801102,1.27181861733,-0.647644532974,0.158469536972,1.99008301633,1.16418756066,0.242660158567,1.37992009705,-0.0545587054529,0.795233949293,0.019089961908,-0.905438136734,0.430271331258,0.934650063047,-0.346101872225,-1.09712188355,-0.528196069148,-2.37977527399,-0.607683691491,-1.0752900903,2.02240506659,-0.564875296813,-1.54292905059,0.870841778933,-0.175210526849,0.0486030066894,0.188646203235,0.209313488475,-0.374444916506,0.954698597251,0.52324766251,-0.495818519936};


    for (int k = 0; k < 1000; k++)
    {

//        float* x = new float[8]{-0.171464609619,-0.94436859941,0.280864675418,0.738247111298,0.650753231022,0.614740629574,-0.126568594674,1.57887424707};
//        Matrix_info info(8);
        float* E = new float[225]{0.441227,-0.33087,2.43077,-0.252092,0.10961,1.58248,-0.909232,-0.591637,0.187603,-0.32987,-1.19276,-0.204877,-0.358829,0.603472,-1.66479,-0.700179,1.15139,1.85733,-1.51118,0.644848,-0.980608,-0.856853,-0.871879,-0.422508,0.99644,0.712421,0.0591442,-0.363311,0.00328884,-0.10593,0.793053,-0.631572,-0.00619491,-0.101068,-0.0523081,0.249218,0.19766,1.33485,-0.0868756,1.56153,-0.305853,-0.477731,0.100738,0.355438,0.269612,1.29196,1.13934,0.49444,-0.336336,-0.100614,1.4134,0.221254,-1.31077,-0.689565,-0.577513,1.1522,-0.107164,2.26011,0.656619,0.124807,-0.435704,0.972179,-0.240711,-0.824123,0.568133,0.0127583,1.18906,-0.0735933,-2.85969,0.789366,-1.87774,1.53876,1.82136,-0.427031,-1.1647,-1.39707,0.872655,-0.202118,-0.59836,-0.24342,2.08851,0.346919,0.745727,0.776908,1.01842,1.06135,-0.710466,-0.215188,-0.76076,-0.711163,1.14151,-0.501756,-0.0791514,-0.692826,-0.593403,0.788238,-0.44543,-0.48212,0.493558,0.500487,0.792423,0.170764,-1.75374,0.630296,0.498329,1.01814,-0.846469,2.52081,-1.23239,0.726953,0.0459552,-0.487133,0.816132,-0.28143,-2.33562,-1.16728,0.457658,2.23797,-1.48126,-0.0169453,1.45073,0.60687,-0.375621,-1.42192,-1.78115,-0.747906,-0.36841,-2.24912,-1.69368,0.303648,-0.408992,-0.754831,-0.407519,-0.812625,0.927516,1.63995,2.07362,0.709798,0.747153,1.4631,1.73845,1.4652,1.21228,-0.634652,-1.5997,0.877153,-0.0938324,-0.055671,-0.889421,-1.30095,1.40217,0.465101,-1.06503,0.390421,0.3056,0.52185,2.23327,-0.0347021,-1.27962,0.0365426,-0.646357,0.548568,0.210542,0.346502,-0.567051,0.413679,-0.510256,0.517259,-0.301005,-1.11841,0.498524,-0.706094,1.44388,0.442956,0.467705,0.101345,-0.059352,-2.3867,1.22217,-0.813912,0.956262,-0.638511,-0.143126,-0.22419,-1.0385,-0.171709,0.476346,-0.414178,-1.26408,-0.573216,0.249817,1.1472,0.835944,0.287404,-0.995596,0.906889,0.0242107,-0.239982,0.910111,0.617845,0.499618,-1.15154,-0.610516,-1.70389,0.194437,0.0282412,0.932561,0.212043,-0.367945,2.11149,-1.02957,-1.33628,-0.610567,0.524694,-0.349308,-0.440738,-1.12129,1.47284,-0.623372,-1.0807,-0.12253,-0.807743,-0.232556,1.33515,-0.446457};
        net.set(E);

        float* x = new float[30]{-0.00497886843115,-0.0368544779081,-0.0191739568452,0.0819679920076,0.0531633724294,-0.0341615035899,-0.0930900480527,-0.0134216994281,0.0832593613024,-0.00173532689521,-0.0127658221954,-0.180791662422,0.0993968976872,-0.149112885882,-0.128210748006,-0.0375707409781,0.00346438757678,0.00450781580839,-0.0763746892846,-0.0313138507197,-0.0606989535576,-0.180955122882,-0.0255517742271,-0.0693799346817,0.041919775764,-0.0145200186562,0.0963801299882,0.0696221988227,0.0899405462376,0.120837807315};
        Matrix_info info(1, 1, 5, 6);

        x = net.forward(x, info);
        for (int i = 0; i < 3; i++)
            cout << x[i] << endl;

        delete [] x;
        delete [] E;
//        net.clean();
    }

    int a = 0;
}


//    for (int i = 0; i < input_channels; i++)
//        for (int j = 0; j < input_height; j++)
//        {
//            for (int k = 0; k < input_width; k++)
//                cout << input_data[i * input_height * input_width + j * input_width + k] << ",  ";
//            cout << endl;
//        }

//for (int i = 0; i < 30; i++)
//cout << input_data[i ] << ",  ";
//
//cout << endl;