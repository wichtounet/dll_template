#include <vector>

#define DLL_SVM_SUPPORT

#define DLL_PARALLEL //Comment this line if you don't want to use multithreading

#include "dll/rbm.hpp"
#include "dll/dbn.hpp"
#include "dll/ocv_visualizer.hpp"
#include "dll/test.hpp"

#include "dll/cpp_utils/data.hpp"

void read_data(std::vector<std::vector<double>>& samples, std::vector<std::size_t>& labels);

void rbm_features();
void svm_classify();

/*
 * The DLL library can be used in two ways:
 * 1. For feature extraction, in which case a RBM (or DBN) is trained on the samples
 *    and the features (probabilities activation) are used for another purpose.
 * 2. For classification. In which case, a DBN is trained on the samples and
 *    fine-tuned with the labels. Fine-tuning can be backpropagation, Conjugate Gradient
 *    or SVM classification on top of the features.
 *
 * For case 1, check rbm_train() and rbm_features()
 * For case 2, check svm_train() and svm_test()
 *
 * Each function use the read_data function, therefore this function should be
 * completed first.
 */

int main(){
	//Call the function you are interested in and complete it

    return 0;
}

//0. Configure the RBM

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

void rbm_train(){
    //1. Create the RBM

    auto rbm = std::make_unique<rbm_t>();

    //2. Read dataset

    std::vector<std::vector<double>> samples;     //All the samples
    std::vector<std::size_t> labels;              //All the labels

    read_data(samples, labels);

    //3. Train the RBM for 100 epochs

    rbm->train(samples, 100);

    //4. Store the RBM

    rbm->store("file.dat");
}

//Extract features from a RBM
void rbm_features(){
    //1. Create the RBM

    auto rbm = std::make_unique<rbm_t>();

    //2. Load from file

    rbm->load("file.dat");

    //2. Read dataset

    std::vector<std::vector<double>> samples;     //All the samples
    std::vector<std::size_t> labels;              //All the labels

    read_data(samples, labels);

    //3. Get the activation probabilities for each sample

    for(auto& sample : samples){
        auto probs = rbm->activation_probabilities(samples[0]);

        //Do something with the extracted features
    }
}

//0. Configure the DBN

using dbn_t = dll::dbn_desc<
    dll::dbn_label_layers<
        dll::rbm_desc<28 * 28, 100, dll::batch_size<50>, dll::init_weights, dll::momentum, dll::weight_decay<dll::decay_type::L2>>::rbm_t,
        dll::rbm_desc<100, 100, dll::batch_size<50>, dll::momentum, dll::weight_decay<dll::decay_type::L2>>::rbm_t,
        dll::rbm_desc<100, 200, dll::batch_size<50>, dll::momentum, dll::weight_decay<dll::decay_type::L2>>::rbm_t
    >
    //, dll::watcher<dll::opencv_dbn_visualizer> //For visualization
    >::dbn_t;

void svm_train(){
    //1. Create the DBN

    auto dbn = std::make_unique<dbn_t>();

    //2. Read dataset

    std::vector<std::vector<double>> samples;     //All the samples
    std::vector<std::size_t> labels;              //All the labels

    read_data(samples, labels);

    //3. Train the DBN layers for x epochs

    dbn->pretrain(samples, 100);

    //4. Train the SVM

    dbn->svm_train(samples, labels);

    //5. Store the file if you want to save it for later

    dbn->store("file.dat"); //Store to file
}

void svm_test(){
    //1. Create the DBN

    auto dbn = std::make_unique<dbn_t>();

    //2. Load the DBN from file

    dbn->load("file.dat");

    //3. Read dataset

    std::vector<std::vector<double>> samples;     //All the samples
    std::vector<std::size_t> labels;              //All the labels

    read_data(samples, labels);

    //4. Compute accuracy on the training set

    auto training_error = dll::test_set(dbn, samples, labels, dll::svm_predictor());

    //5. Example: Get the predicted label for each sample

    for(std::size_t i = 0; i < samples.size(); ++i){
        const auto& sample = samples[i];
        const auto& label = labels[i];

        auto predicted = dbn->svm_predict(sample);

        //Do something with the predicted label
    }
}

void read_data(std::vector<std::vector<double>>& samples, std::vector<std::size_t>& labels){
    //TODO Read samples
    //TODO Read labels

    //cpp::normalize_each(samples); //For gaussian visible units
}
