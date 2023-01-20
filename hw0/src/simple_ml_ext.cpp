#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


float * matrix_multip(const float * mat1, size_t rows1, size_t cols1, 
                      float * mat2, size_t rows2, size_t cols2){
    float *res = new float [rows1 * cols2]();
    for(size_t i=0; i < rows1; i++)
        for(size_t j=0; j < cols2; j++)
            for (size_t k=0; k < cols1; k++)
                res[i*cols2+ j] += mat1[i*cols1+k] * mat2[k*cols2+j];
    return res;                        
}


float * matrix_minus(float *mat1, float *mat2, size_t rows, size_t cols){
    size_t length = cols * rows;
    float * res = new float [length]();
    for (size_t i=0; i < length; i++)
        res[i] = mat1[i] - mat2[i];
    return res;
}


void softmax_normize(float *Z, size_t rows, size_t cols){
    for (size_t i = 0; i<rows; i++){
        double cur_row_sum = 0.0;

        for (size_t j=0; j < cols; j++){
            Z[i * cols + j] = exp(Z[i * cols + j]);
            cur_row_sum += Z[i * cols + j];
        }
        
        for (size_t j=0; j < cols; j++)
            Z[i * cols + j] /= cur_row_sum;
    }    
}


float * build_I(size_t rows, size_t cols, const unsigned char * y){
    float * I = new float[rows * cols]();
    for (size_t i=0; i<rows; i++)
        I[i * cols + y[i]] = 1;
    return I;
}


void update_theta(float *theta, float *dtheta, size_t rows, size_t cols, float lr, size_t batch){
    size_t length = rows * cols;
    for (size_t i=0; i<length; i++)
        theta[i] -= lr / batch * dtheta[i];
}


float * transpose_mat(const float * X, size_t rows, size_t cols){
    float * XT = new float[cols * rows]();
    for(size_t i=0; i<rows; i++)
        for(size_t j=0; j<cols; j++)
            XT[j*rows + i] = X[i * cols + j];
    return XT;
}


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    // forward

    //const float * X_ = X;
    //const unsigned char *y_ = y;
    //batch = m;

    for(size_t i=0; i < m; i+=batch){
        const float * X_ = &X[i*n];
        const unsigned char * y_ = &y[i];
    
        float *Z = matrix_multip(X_, batch, n, theta, n, k);
        softmax_normize(Z, batch, k);
        //std::cout<<"forward"<<std::endl;
        //backward
        float *XT = transpose_mat(X_, batch, n);
        float *I = build_I(batch, k, y_);

        float *Z_I = matrix_minus(Z, I, batch, k);

        float *dtheta = matrix_multip(XT, n, batch, Z_I, batch, k);
        update_theta(theta, dtheta, n, k, lr, batch);

        delete [] Z;
        delete [] XT;
        delete [] Z_I;
        delete [] I;
        delete [] dtheta;
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
