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

    float *v; // units activations, sampled
    float *h;

    float *v_prob; // units activation probabilities
    float *h_prob;

    float *b;   // weights
    float *c;
    float *W;

    // init units and weights randomly, in some reasonable ranges
    void randomInit();

    // stochastic sampling step
    void update();

    // images for visualization
    void makeImages();
    vector<ofImage *> filters;
    ofImage *v_image;
    ofImage *h_image;
    ofImage *v_prob_image;
    ofImage *h_prob_image;
};


