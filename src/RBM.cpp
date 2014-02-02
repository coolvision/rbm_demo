/*
 * RBM.cpp
 *
 *  Created on: Jan 8, 2014
 *      Author: sk
 */

#include "RBM.h"

void RBM::setTrainMode(int train_method, int k) {

    if (train_method == 0) {
        pcd_on = false;
    } else {
        pcd_on = true;
    }

    this->k = k;
}

void RBM::init(int image_sqr, int n_hidden_sqr,
        float* data, float *labels, int n_samples, int batch_size) {

    training_data = data;
    this->labels = labels;
    n_training_samples = n_samples;

    learning_rate = 0.05;
    momentum = 0.5f;
    weightcost = 0.0002;
    pcd_on = false;
    k = 5;

    inhibit_sparsity = false;
    sparsity_k = 0.01f;
    sparsity_target = 0.1f;
    inhibit_selectivity = false;
    selectivity_k = 0.01;
    selectivity_target = 0.1f;

    sample_i = 0;
    epoch_i = 0;
    batch_i = 0;
    sample_offset = training_data;
    this->batch_size = batch_size;

    image_side = image_sqr;
    h_image_side = n_hidden_sqr;
    n_visible = image_side * image_side + 1;    // additional bias unit
    n_hidden = h_image_side * h_image_side + 1;

    allocate();

    // set biases to 0
    //memset(b, 0, n_visible * sizeof(float));
    //memset(c, 0, n_hidden * sizeof(float));
    memset(b_inc, 0, n_visible * sizeof(float));
    memset(c_inc, 0, n_hidden * sizeof(float));
    memset(W_inc, 0, n_hidden * n_visible * sizeof(float));

    memset(mean_activity, 0, n_hidden * sizeof(float));
    q = 0.0f;

    for (int i = 0; i < n_visible * n_hidden; i++) {
        W[i] = randn(0.0, 0.01);
    }
}

void RBM::allocate() {

    // allocate memory for units and weights
    v_data = new float[n_visible * batch_size];
    h_data_prob = new float[n_hidden * batch_size];
    h_data = new float[n_hidden * batch_size];

    v_prob = new float[n_visible * batch_size];
    v = new float[n_visible * batch_size];
    h_prob = new float[n_hidden * batch_size];
    h = new float[n_hidden * batch_size];

    // store transposed data matrix
    vt = new float[n_visible * batch_size];

    mean_activity = new float[n_hidden];

    W = new float[n_visible * n_hidden];
    Wt = new float[n_visible * n_hidden]; // hold transposed weights
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

void RBM::release() {

    delete[] v_data;

    delete[] h_data_prob;
    delete[] h_data;

    delete[] v_prob;
    delete[] v;

    delete[] h_prob;
    delete[] h;

    delete[] mean_activity;

    delete[] W;
    delete[] Wt;
    delete[] pos_weights;
    delete[] neg_weights;

    delete[] b_inc;
    delete[] c_inc;
    delete[] W_inc;
}


RBM::RBM() {

}

RBM::~RBM() {

    release();
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


