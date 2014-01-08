/*
 * RBMDraw.cpp
 *
 *  Created on: Jan 8, 2014
 *      Author: sk
 */

#include "RBM.h"

void RBM::makeImages() {

    // put units values and weights into images

    // filters from weights
    for (int i = 0; i < n_hidden; i++) {

        float min_v = FLT_MAX;
        float max_v = -FLT_MAX;

        for (int j = 0; j < n_visible; j++) {
            int w_i = i * n_visible + j;
            if (W[w_i] > max_v)
                max_v = W[w_i];
            if (W[w_i] < min_v)
                min_v = W[w_i];
        }

        for (int j = 0; j < n_visible; j++) {
            int w_i = i * n_visible + j;

            if (W[w_i] > 0) {
                filters[i]->setColor(j % image_side, j / image_side,
                        ofColor(((W[w_i] - min_v) / (max_v - min_v)) * 255.0f));
            } else {
                filters[i]->setColor(j % image_side, j / image_side,
                        ofColor(((W[w_i] - min_v) / (max_v - min_v)) * 255.0f,
                                0.0f, 0.0f));
            }
        }

        filters[i]->update();
    }

    // activation probabilities and sampled activations
    for (int i = 0; i < n_visible; i++) {
        v_data_image->setColor(i % image_side, i / image_side,
                ofColor(v_data[i] * 255.0f));
        v_n_prob_image->setColor(i % image_side, i / image_side,
                ofColor(v_negative_prob[i] * 255.0f));
        v_n_image->setColor(i % image_side, i / image_side,
                ofColor(v_negative[i] * 255.0f));
    }

    for (int i = 0; i < n_hidden; i++) {
        h_image->setColor(i % h_image_side, i / h_image_side,
                ofColor(h[i] * 255.0f));
        h_n_image->setColor(i % h_image_side, i / h_image_side,
                ofColor(h_negative[i] * 255.0f));
    }

    float max_v;
    max_v = -FLT_MAX;
    for (int i = 0; i < n_hidden; i++) {
        if (h_prob[i] > max_v) {
            max_v = h_prob[i];
        }
    }
    for (int i = 0; i < n_hidden; i++) {
        h_prob_image->setColor(i % h_image_side, i / h_image_side,
                ofColor((h_prob[i] / max_v) * 255.0f));
    }

    max_v = -FLT_MAX;
    for (int i = 0; i < n_hidden; i++) {
        if (h_negative_prob[i] > max_v) {
            max_v = h_negative_prob[i];
        }
    }
    for (int i = 0; i < n_hidden; i++) {
        h_n_prob_image->setColor(i % h_image_side, i / h_image_side,
                ofColor((h_prob[i] / max_v) * 255.0f));
    }

    v_data_image->update();
    v_n_prob_image->update();
    v_n_image->update();

    h_prob_image->update();
    h_image->update();
    h_n_prob_image->update();
    h_n_image->update();
}
