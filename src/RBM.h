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

    // units from data
    float *v_data;
    float *h_data_prob;
    float *h_data;

    // running sampling
    float *v_prob;  // for CD1, set to data, run the chain
    float *v;       // for PCD, run the chain independently

    float *h_prob;
    float *h;

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
    ofImage *h_data_prob_image;
    ofImage *h_data_image;

    ofImage *v_prob_image;
    ofImage *v_image;

    ofImage *h_prob_image;
    ofImage *h_image;
};


