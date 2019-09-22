//
// Created by xuewei on 2019/9/19.
//

#ifndef VLEARN_UTILS_H
#define VLEARN_UTILS_H
#include <vector>

typedef float real;

namespace vlearn{
    namespace util {
        real sigmoid(real x);

        real inner_product(std::vector<real>& a, int x_start, std::vector<real>& b, int y_start, int m);

        real accuracy(std::vector<int>& y, std::vector<int>& pred);
    }
}
#endif //VLEARN_UTILS_H
