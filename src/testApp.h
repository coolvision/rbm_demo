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

    // run stochastic sampling for some number of iterations
    void gibbsStep(int n);

    // images for visualization
    void makeImages();
    vector<ofImage *> filters;
    ofImage *v_image;
    ofImage *h_image;
    ofImage *v_prob_image;
    ofImage *h_prob_image;
};

class testApp : public ofBaseApp{

    vector<ofImage *> images;

    RBM *rbm;

	public:
		void setup();
		void update();
		void draw();

		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);
		
};
