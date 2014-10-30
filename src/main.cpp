#include <vector>

#define DLL_SVM_SUPPORT

#include "dll/rbm.hpp"
#include "dll/dbn.hpp"
#include "dll/ocv_visualizer.hpp"


void read_data(std::vector<std::vector<double>>& samples, std::vector<std::size_t>& labels);

void rbm_features();
void svm_classify();

int main(){
    //rbm_features(); //Only one RBM, no classification, pure feature extraction
    //svm_classify(); //Classify RBM features with SVM

    return 0;
}

void rbm_features(){
    //1. Configure and create the RBM

    using rbm_t = dll::rbm_desc<
       28 * 28,                                             //Number of visible units
       200                                                  //Number of hidden units
       , dll::momentum                                      //Activate momentum ?
       , dll::batch_size<25>                                //Minibatch
       , dll::weight_decay<>                                //Activate weight decay ?
       //, dll::sparsity<>                                  //Activate weight decay ?
       //, dll::visible<dll::unit_type::GAUSSIAN>           //Gaussian visible units ?
       //, dll::watcher<dll::opencv_rbm_visualizer>         //OpenCV Visualizer ?
    >::rbm_t;

    auto rbm = std::make_unique<rbm_t>();

    rbm->load("file.dat") //Load from file

    //2. Read dataset

    std::vector<std::vector<double>> samples;     //All the samples
    std::vector<std::size_t> labels;              //All the labels

    read_data(samples, labels);

    //3. Train the RBM for x epochs

    rbm->train(samples, 100);

    //4. Get the activation probabilities for a sample

    auto probs = rbm->activation_probabilities(samples[0]);

    rbm->store("file.dat") //Store to file
}

void svm_classify(){
    //1. Configure and create the RBM

    using dbn_t = dll::dbn_desc<
        dll::dbn_label_layers<
            dll::rbm_desc<28 * 28, 100, dll::batch_size<50>, dll::init_weights, dll::momentum, dll::weight_decay<dll::decay_type::L2>>::rbm_t,
            dll::rbm_desc<100, 100, dll::batch_size<50>, dll::momentum, dll::weight_decay<dll::decay_type::L2>>::rbm_t,
            dll::rbm_desc<100, 200, dll::batch_size<50>, dll::momentum, dll::weight_decay<dll::decay_type::L2>>::rbm_t
        >>::dbn_t;

    auto dbn = std::make_unique<dbn_t>();

    dbn->load("file.dat"); //Load from file

    //2. Read dataset

    std::vector<std::vector<double>> samples;     //All the samples
    std::vector<std::size_t> labels;              //All the labels

    read_data(samples, labels);

    //3. Train the DBN layers for x epochs

    dbn->pretrain(samples, 100);

    //4. Get the activation probabilities for a sample

    auto probs = dbn->activation_probabilities(samples[0]);

    //5. Train the SVM

    dbn->svm_train(samples, labels);

    //6. Predict the label of a sample

    auto label = dbn->svm_predict(samples[0]);

    dbn->store("file.dat"); //Store to file
}

void read_data(std::vector<std::vector<double>>& samples, std::vector<std::size_t>& labels){
    //TODO Read samples
    //TODO Read labels
}
