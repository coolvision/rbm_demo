/*
 * RBM.h
 *
 *  Created on: Jan 8, 2014
 *      Author: sk
 */

#pragma once

#include "ofMain.h"

class RBM {
public:

    RBM(int image_side, int n_hidden);
    ~RBM();

    // initially, used only for rectangular images
    int image_side;
    int h_image_side; // image used for hidden units visualization

    // number of visible and hidden units
    int n_visible;
    int n_hidden;

    // visible units from data
    float *v_data;

    // hidden nodes, positive phase
    float *h_prob;
    float *h;

    // negarive phase
    float *v_negative_prob; // units activation probabilities
    float *v_negative; // units activations, sampled

    float *h_negative_prob;
    float *h_negative;

    float *b;   // weights
    float *c;
    float *W;
    float *pos_weights; // for gradient approximation
    float *neg_weights;

    // init units and weights randomly, in some reasonable ranges
    void randomInit();

    // stochastic sampling step
    void update();

    // images for visualization
    void makeImages();

    vector<ofImage *> filters;
    ofImage *v_bias;
    ofImage *h_bias;

    ofImage *v_data_image;

    ofImage *h_prob_image;
    ofImage *h_image;

    ofImage *v_n_prob_image;
    ofImage *v_n_image;

    ofImage *h_n_prob_image;
    ofImage *h_n_image;
};


