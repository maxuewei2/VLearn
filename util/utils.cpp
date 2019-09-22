//
// Created by xuewei on 2019/9/19.
//
#include "utils.h"
#include <cmath>

namespace vlearn{
    namespace util {
        real sigmoid(real x) {
            return 1.0 / (1.0 + exp(-x));
        }

        real inner_product(std::vector<real>& a, int x_start, std::vector<real>& b, int y_start, int m) {
            real r = 0;
            for (int i = 0; i < m; i++) {
                r += (a[x_start+i] * b[y_start+i]);
            }
            return r;
        }
        real accuracy(std::vector<int>& y, std::vector<int>& pred) {
            int sum = 0;
            int n=y.size();
            for (int i = 0; i < n; i++) {
                if (y[i] == pred[i]) {
                    sum++;
                }
            }
            return (real) sum / (real) n;
        }
    }
}
