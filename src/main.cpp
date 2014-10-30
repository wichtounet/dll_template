#include <vector>

#include "dll/rbm.hpp"
#include "dll/ocv_visualizer.hpp"

int main(){
    //1. Configure and create the RBM

    using rbm_t = dll::rbm_desc<
       28 * 28,                       //Number of visible units
       200                            //Number of hidden units
       , dll::momentum                //Activate momentum ?
       , dll::batch_size<25>          //Minibatch
       , dll::weight_decay<>          //Activate weight decay ?
       //, dll::sparsity<>              //Activate weight decay ?
       //, dll::visible<dll::unit_type::GAUSSIAN>  //Gaussian visible units ?
       //, dll::watcher<dll::opencv_rbm_visualizer>  //OpenCV Visualizer ?
    >::rbm_t;

    auto rbm = std::make_unique<rbm_t>();

    //2. Read dataset

    std::vector<std::vector<double>> samples;     //All the samples
    std::vector<std::size_t> labels;              //All the labels

    //TODO Read samples
    //TODO Read labels

    //3. Train the RBM for x epochs

    rbm->train(samples, 100);

    //4. Get the activation probabilities for a sample

    auto probs = rbm->activation_probabilities(samples[0]);
}
