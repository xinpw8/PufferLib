#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// File format is obained by flattening and concatenating all pytorch layers
typedef struct Weights Weights;
struct Weights {
    float* data;
    int size;
    int idx;
};

float* _load_weights(const char* filename, int num_weights) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    rewind(file);
    float* weights = calloc(num_weights, sizeof(float));
    int read_size = fread(weights, sizeof(float), num_weights, file);
    fclose(file);
    if (read_size != num_weights) {
        perror("Error reading file");
        return NULL;
    }
    return weights;
}

Weights* load_weights(const char* filename, int num_weights) {
    Weights* weights = calloc(1, sizeof(Weights));
    weights->data = _load_weights(filename, num_weights);
    weights->size = num_weights;
    weights->idx = 0;
    return weights;
}

void free_weights(Weights* weights) {
    free(weights->data);
    free(weights);
}

float* get_weights(Weights* weights, int num_weights) {
    float* data = &weights->data[weights->idx];
    weights->idx += num_weights;
    assert(weights->idx <= weights->size);
    return data;
}

// PufferNet implementation of PyTorch functions
// These are tested against the PyTorch implementation
void _relu(float* input, float* output, int size) {
    for (int i = 0; i < size; i++)
        output[i] = fmaxf(0.0f, input[i]);
}

float _sigmoid(float x);
inline float _sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void _linear(float* input, float* weights, float* bias, float* output,
        int batch_size, int input_dim, int output_dim) {
    for (int b = 0; b < batch_size; b++) {
        for (int o = 0; o < output_dim; o++) {
            float sum = 0.0f;
            for (int i = 0; i < input_dim; i++)
                sum += input[b*input_dim + i] * weights[o*input_dim + i];
            output[b*output_dim + o] = sum + bias[o];
        }
    }
}

void _linear_accumulate(float* input, float* weights, float* bias, float* output,
        int batch_size, int input_dim, int output_dim) {
    for (int b = 0; b < batch_size; b++) {
        for (int o = 0; o < output_dim; o++) {
            float sum = 0.0f;
            for (int i = 0; i < input_dim; i++)
                sum += input[b*input_dim + i] * weights[o*input_dim + i];
            output[b*output_dim + o] += sum + bias[o];
        }
    }
}

void _conv2d(float* input, float* weights, float* bias,
        float* output, int batch_size, int in_width, int in_height,
        int in_channels, int out_channels, int kernel_size, int stride) {
    int h_out = (in_height - kernel_size)/stride + 1;
    int w_out = (in_width - kernel_size)/stride + 1;
    for (int b = 0; b < batch_size; b++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int h = 0; h < h_out; h++) {
                for (int w = 0; w < w_out; w++) {
                    int out_adr = (
                        b*out_channels*h_out*w_out
                        + oc*h_out*w_out+ 
                        + h*w_out
                        + w
                    );
                    output[out_adr] = bias[oc];
                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int kh = 0; kh < kernel_size; kh++) {
                            for (int kw = 0; kw < kernel_size; kw++) {
                                int in_adr = (
                                    b*in_channels*in_height*in_width
                                    + ic*in_height*in_width
                                    + (h*stride + kh)*in_width
                                    + (w*stride + kw)
                                );
                                int weight_adr = (
                                    oc*in_channels*kernel_size*kernel_size
                                    + ic*kernel_size*kernel_size
                                    + kh*kernel_size
                                    + kw
                                );
                                output[out_adr] += input[in_adr]*weights[weight_adr];
                            }
                        }
                    }
               }
            }
        }
    }
}

void _lstm(float* input, float* state_h, float* state_c, float* weights_input,
        float* weights_state, float* bias_input, float*bias_state,
        float *buffer, int batch_size, int input_size, int hidden_size) {
    _linear(input, weights_input, bias_input, buffer, batch_size, input_size, 4*hidden_size);
    _linear_accumulate(state_h, weights_state, bias_state, buffer, batch_size, hidden_size, 4*hidden_size);

    // Activation functions
    for (int b=0; b<batch_size; b++) {
        int b_offset = 4*b*hidden_size;
        for (int i=0; i<2*hidden_size; i++) {
            int buf_adr = b_offset + i;
            buffer[buf_adr] = _sigmoid(buffer[buf_adr]);
        }
        for (int i=2*hidden_size; i<3*hidden_size; i++) {
            int buf_adr = b_offset + i;
            buffer[buf_adr] = tanh(buffer[buf_adr]);
        }
        for (int i=3*hidden_size; i<4*hidden_size; i++) {
            int buf_adr = b_offset + i;
            buffer[buf_adr] = _sigmoid(buffer[buf_adr]);
        }
    }

    // Gates
    for (int b=0; b<batch_size; b++) {
        int hidden_offset = b*hidden_size;
        int b_offset = 4*b*hidden_size;
        for (int i=0; i<hidden_size; i++) {
            state_c[hidden_offset + i] = (
                buffer[b_offset + hidden_size + i] * state_c[hidden_offset + i]
                + buffer[b_offset + i] * buffer[b_offset + 2*hidden_size + i]
            );
            state_h[hidden_offset + i] = (
                buffer[b_offset + 3*hidden_size + i] * tanh(state_c[hidden_offset + i])
            );
        }
    }
}

void _one_hot(int* input, int* output, int batch_size, int input_size, int num_classes) {
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < input_size; i++) {
            int in_adr = b*input_size + i;
            int out_adr = (
                b*input_size*num_classes
                + i*num_classes
                + input[in_adr]
            );
            output[out_adr] = 1.0f;
        }
    }
}

void _cat_dim1(float* x, float* y, float* output, int batch_size, int x_size, int y_size) {
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < x_size; i++) {
            int x_adr = b*x_size + i;
            int out_adr = b*(x_size + y_size) + i;
            output[out_adr] = x[x_adr];
        }
        for (int i = 0; i < y_size; i++) {
            int y_adr = b*y_size + i;
            int out_adr = b*(x_size + y_size) + x_size + i;
            output[out_adr] = y[y_adr];
        }
    }
}

void _argmax_multidiscrete(float* input, int* output, int batch_size, int logit_sizes[], int num_actions) {
    int in_adr = 0;
    for (int b = 0; b < batch_size; b++) {
        for (int a = 0; a < num_actions; a++) {
            int out_adr = b*num_actions + a;
            float max_logit = input[in_adr];
            output[out_adr] = 0;
            int num_action_types = logit_sizes[a];
            for (int i = 1; i < num_action_types; i++) {
                float out = input[in_adr + i];
                if (out > max_logit) {
                    max_logit = out;
                    output[out_adr] = i;
                }
            }
            in_adr += num_action_types;
        }
    }
}

void _softmax_multidiscrete(float* input, int* output, int batch_size, int logit_sizes[], int num_actions) {
    int in_adr = 0;
    for (int b = 0; b < batch_size; b++) {
        for (int a = 0; a < num_actions; a++) {
            int out_adr = b*num_actions + a;
            float logit_exp_sum = 0;
            int num_action_types = logit_sizes[a];
            for (int i = 0; i < num_action_types; i++) {
                logit_exp_sum += expf(input[in_adr + i]);
            }
            float prob = rand() / (float)RAND_MAX;
            bool found = false;
            float logit_prob = 0;
            for (int i = 0; i < num_action_types; i++) {
                logit_prob += expf(input[in_adr + i]) / logit_exp_sum;
                if (prob < logit_prob) {
                    output[out_adr] = i;
                    found = true;
                    break;
                }
            }
            assert(found);
            in_adr += num_action_types;
        }
    }
}

// User API. Provided to help organize layers
typedef struct Linear Linear;
struct Linear {
    float* output;
    float* weights;
    float* bias;
    int batch_size;
    int input_dim;
    int output_dim;
};

Linear* make_linear(Weights* weights, int batch_size, int input_dim, int output_dim) {
    Linear* layer = calloc(1, sizeof(Linear));
    layer->output = calloc(batch_size*output_dim, sizeof(float));
    layer->weights = get_weights(weights, output_dim*input_dim);
    layer->bias = get_weights(weights, output_dim);
    layer->batch_size = batch_size;
    layer->input_dim = input_dim;
    layer->output_dim = output_dim;
    return layer;
}

void free_linear(Linear* layer) {
    free(layer->output);
    free(layer);
}

void linear(Linear* layer, float* input) {
    _linear(input, layer->weights, layer->bias, layer->output,
        layer->batch_size, layer->input_dim, layer->output_dim);
}

void linear_accumulate(Linear* layer, float* input) {
    _linear_accumulate(input, layer->weights, layer->bias, layer->output,
        layer->batch_size, layer->input_dim, layer->output_dim);
}

typedef struct ReLU ReLU;
struct ReLU {
    float* output;
    int batch_size;
    int input_dim;
};

ReLU* make_relu(int batch_size, int input_dim) {
    ReLU* layer = calloc(1, sizeof(ReLU));
    layer->output = calloc(batch_size*input_dim, sizeof(float));
    layer->batch_size = batch_size;
    layer->input_dim = input_dim;
    return layer;
}

void free_relu(ReLU* layer) {
    free(layer->output);
    free(layer);
}

void relu(ReLU* layer, float* input) {
    _relu(input, layer->output, layer->batch_size*layer->input_dim);
}

typedef struct Conv2D Conv2D;
struct Conv2D {
    float* output;
    float* weights;
    float* bias;
    int batch_size;
    int in_width;
    int in_height;
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
};

Conv2D* make_conv2d(Weights* weights, int batch_size, int in_width, int in_height,
        int in_channels, int out_channels, int kernel_size, int stride) {
    Conv2D* layer = calloc(1, sizeof(Conv2D));
    layer->output = calloc(batch_size*out_channels*in_height*in_width, sizeof(float));
    layer->weights = get_weights(weights, out_channels*in_channels*kernel_size*kernel_size);
    layer->bias = get_weights(weights, out_channels);
    layer->batch_size = batch_size;
    layer->in_width = in_width;
    layer->in_height = in_height;
    layer->in_channels = in_channels;
    layer->out_channels = out_channels;
    layer->kernel_size = kernel_size;
    layer->stride = stride;
    return layer;
}

void free_conv2d(Conv2D* layer) {
    free(layer->output);
    free(layer);
}

void conv2d(Conv2D* layer, float* input) {
    _conv2d(input, layer->weights, layer->bias, layer->output,
        layer->batch_size, layer->in_width, layer->in_height,
        layer->in_channels, layer->out_channels, layer->kernel_size, layer->stride);
}

typedef struct LSTM LSTM;
struct LSTM {
    float* state_h;
    float* state_c;
    float* weights_input;
    float* weights_state;
    float* bias_input;
    float*bias_state;
    float *buffer;
    int batch_size;
    int input_size;
    int hidden_size;
};

LSTM* make_lstm(Weights* weights, int batch_size, int input_size, int hidden_size) {
    LSTM* layer = calloc(1, sizeof(LSTM));
    layer->state_h = calloc(batch_size*hidden_size, sizeof(float));
    layer->state_c = calloc(batch_size*hidden_size, sizeof(float));
    layer->weights_input = get_weights(weights, 4*hidden_size*input_size);
    layer->weights_state = get_weights(weights, 4*hidden_size*hidden_size);
    layer->bias_input = get_weights(weights, 4*hidden_size);
    layer->bias_state = get_weights(weights, 4*hidden_size);
    layer->buffer = calloc(4*batch_size*hidden_size, sizeof(float));
    layer->batch_size = batch_size;
    layer->input_size = input_size;
    layer->hidden_size = hidden_size;
    return layer;
}

void free_lstm(LSTM* layer) {
    free(layer->state_h);
    free(layer->state_c);
    free(layer->buffer);
    free(layer);
}

void lstm(LSTM* layer, float* input) {
    _lstm(input, layer->state_h, layer->state_c, layer->weights_input,
        layer->weights_state, layer->bias_input, layer->bias_state,
        layer->buffer, layer->batch_size, layer->input_size, layer->hidden_size);
}

typedef struct OneHot OneHot;
struct OneHot {
    int* output;
    int batch_size;
    int input_size;
    int num_classes;
};

OneHot* make_one_hot(int batch_size, int input_size, int num_classes) {
    OneHot* layer = calloc(1, sizeof(OneHot));
    layer->output = calloc(batch_size*input_size*num_classes, sizeof(int));
    layer->batch_size = batch_size;
    layer->input_size = input_size;
    layer->num_classes = num_classes;
    return layer;
}

void free_one_hot(OneHot* layer) {
    free(layer->output);
    free(layer);
}

void one_hot(OneHot* layer, int* input) {
    _one_hot(input, layer->output, layer->batch_size, layer->input_size, layer->num_classes);
}

typedef struct CatDim1 CatDim1;
struct CatDim1 {
    float* output;
    int batch_size;
    int x_size;
    int y_size;
};

CatDim1* make_cat_dim1(int batch_size, int x_size, int y_size) {
    CatDim1* layer = calloc(1, sizeof(CatDim1));
    layer->output = calloc(batch_size*(x_size + y_size), sizeof(float));
    layer->batch_size = batch_size;
    layer->x_size = x_size;
    layer->y_size = y_size;
    return layer;
}

void free_cat_dim1(CatDim1* layer) {
    free(layer->output);
    free(layer);
}

void cat_dim1(CatDim1* layer, float* x, float* y) {
    _cat_dim1(x, y, layer->output, layer->batch_size, layer->x_size, layer->y_size);
}

typedef struct Multidiscrete Multidiscrete;
struct Multidiscrete {
    int batch_size;
    int logit_sizes[32];
    int num_actions;
};

Multidiscrete* make_multidiscrete(int batch_size, int logit_sizes[], int num_actions) {
    Multidiscrete* layer = calloc(1, sizeof(Multidiscrete));
    layer->batch_size = batch_size;
    layer->num_actions = num_actions;
    memcpy(layer->logit_sizes, logit_sizes, num_actions*sizeof(int));
    return layer;
}

void free_multidiscrete(Multidiscrete* layer) {
    free(layer);
}

void argmax_multidiscrete(Multidiscrete* layer, float* input, int* output) {
    _argmax_multidiscrete(input, output, layer->batch_size, layer->logit_sizes, layer->num_actions);
}

void softmax_multidiscrete(Multidiscrete* layer, float* input, int* output) {
    _softmax_multidiscrete(input, output, layer->batch_size, layer->logit_sizes, layer->num_actions);
}
