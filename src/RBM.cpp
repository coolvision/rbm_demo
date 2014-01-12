/*
 * RBM.cpp
 *
 *  Created on: Jan 8, 2014
 *      Author: sk
 */

#include "RBM.h"

RBM::RBM(int image_side, int n_hidden) {

    this->image_side = image_side;
    this->n_visible = image_side * image_side;
    this->n_hidden = n_hidden;

    // allocate memory for units and weights
    v_data = new float[n_visible];

    h_data_prob = new float[n_hidden];
    h_data = new float[n_hidden];

    v_prob = new float[n_visible];
    v = new float[n_visible];

    h_prob = new float[n_hidden];
    h = new float[n_hidden];

    b = new float[n_visible];
    c = new float[n_hidden];

    mean_activity = new float[n_hidden];
    mean_weight = new float[n_hidden];

    W = new float[n_visible * n_hidden];
    pos_weights = new float[n_visible * n_hidden];
    neg_weights = new float[n_visible * n_hidden];

    b_inc = new float[n_visible];
    c_inc = new float[n_hidden];
    W_inc = new float[n_visible * n_hidden];

    filters.resize(n_hidden);
    for (int i = 0; i < n_hidden; i++) {
        filters[i] = new ofImage();
        filters[i]->allocate(image_side, image_side, OF_IMAGE_COLOR);
    }

    h_image_side = sqrt(n_hidden);

    v_bias = new ofImage();
    v_bias->allocate(image_side, image_side, OF_IMAGE_COLOR);
    h_bias = new ofImage();
    h_bias->allocate(h_image_side, h_image_side, OF_IMAGE_COLOR);

    v_data_image = new ofImage();
    v_data_image->allocate(image_side, image_side, OF_IMAGE_GRAYSCALE);

    h_data_prob_image = new ofImage();
    h_data_prob_image->allocate(h_image_side, h_image_side, OF_IMAGE_GRAYSCALE);
    h_data_image = new ofImage();
    h_data_image->allocate(h_image_side, h_image_side, OF_IMAGE_GRAYSCALE);

    v_prob_image = new ofImage();
    v_prob_image->allocate(image_side, image_side, OF_IMAGE_GRAYSCALE);
    v_image = new ofImage();
    v_image->allocate(image_side, image_side, OF_IMAGE_GRAYSCALE);

    h_prob_image = new ofImage();
    h_prob_image->allocate(h_image_side, h_image_side, OF_IMAGE_GRAYSCALE);
    h_image = new ofImage();
    h_image->allocate(h_image_side, h_image_side, OF_IMAGE_GRAYSCALE);
}

RBM::~RBM() {

    delete[] v_data;

    delete[] h_data_prob;
    delete[] h_data;

    delete[] v_prob;
    delete[] v;

    delete[] h_prob;
    delete[] h;

    delete[] b;
    delete[] c;

    delete[] mean_activity;
    delete[] mean_weight;

    delete[] W;
    delete[] pos_weights;
    delete[] neg_weights;

    delete[] b_inc;
    delete[] c_inc;
    delete[] W_inc;
}

double randn(double mu, double sigma) {

    static bool deviateAvailable = false;
    static float storedDeviate; // deviate from previous calculation
    double polar, rsquared, var1, var2;

    // If no deviate has been stored, the polar Box-Muller transformation is
    // performed, producing two independent normally-distributed random
    // deviates.  One is stored for the next round, and one is returned.
    if (!deviateAvailable) {

        // choose pairs of uniformly distributed deviates, discarding those
        // that don't fall within the unit circle
        do {
            var1 = 2.0 * (double(rand()) / double(RAND_MAX)) - 1.0;
            var2 = 2.0 * (double(rand()) / double(RAND_MAX)) - 1.0;
            rsquared = var1 * var1 + var2 * var2;
        } while (rsquared >= 1.0 || rsquared == 0.0);

        // calculate polar tranformation for each deviate
        polar = sqrt(-2.0 * log(rsquared) / rsquared);

        // store first deviate and set flag
        storedDeviate = var1 * polar;
        deviateAvailable = true;

        // return second deviate
        return var2 * polar * sigma + mu;
    }

    // If a deviate is available from a previous call to this function, it is
    // returned, and the flag is set to false.
    else {
        deviateAvailable = false;
        return storedDeviate * sigma + mu;
    }
}

void RBM::randomInit() {

    // set biases to 0
    memset(b, 0, n_visible * sizeof(float));
    memset(c, 0, n_hidden * sizeof(float));
    memset(b_inc, 0, n_visible * sizeof(float));
    memset(c_inc, 0, n_hidden * sizeof(float));
    memset(W_inc, 0, n_hidden * n_visible * sizeof(float));

    memset(mean_activity, 0, n_hidden * sizeof(float));
    memset(mean_weight, 0, n_hidden * sizeof(float));

    q = 0.0f;

    for (int i = 0; i < n_visible * n_hidden; i++) {
        W[i] = randn(0.0, 0.01);
    }
}

