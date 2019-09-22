#include <cmath>
#include <vector>
#include "classifier.h"

namespace vlearn{
    namespace model {
        class LR : public Classifier {
        protected:
            struct Param{
                int max_iter = 1000;
                real alpha = 0.01;
            };
            Param param;
            std::vector<real> theta;
        public:
            LR(LR &lr) : Classifier(lr) {
                param=lr.param;
            }
            explicit LR(int max_iter = 100000, real alpha = 0.001) {
                param.max_iter = max_iter;
                param.alpha = alpha;
            }
            LR *__clone__() override {
                LR *lr = new LR(*this);
                return lr;
            }

            void __delete__() override {}

            std::vector<real>& get_coef_(){
                return theta;
            }

            void fit(std::vector<real>& X,
                    std::vector<int>& y,
                    int n,
                    int m) override {
                // n samples, m dim
                printf("fitting Logistic Regression with maxiter=%d alpha=%f\n",param.max_iter,param.alpha);
                for(int i=0;i<10;i++) {
                    for (int j = 0; j < m; j++) {
                        printf(" %.5f", X[i * m + j]);
                    }
                    printf("\n");
                }
                theta.resize(m+1);
                for (int i = 0; i < m+1; i++)theta[i] = 0;//(real)rand() / (real)RAND_MAX;
                for (int iter = 0; iter < param.max_iter; iter++) {
                    int r = rand() % n;
                    real inp=util::inner_product(X, r * m, theta, 0, m);
                    inp+=(theta[m]);
                    real g = (real) y[r] - util::sigmoid(inp);
                    g *= param.alpha;
                    for (int i = 0; i < m; i++)theta[i] +=  (X[r * m + i] * g-2*g*theta[i]);//+2*g*theta[i]/(real)n
                    theta[m]+=(g-2*g*theta[m]);//+2*g*theta[m]/(real)n
                }
            }

            std::vector<real> get_pred_prob(std::vector<real>& X,
                             int n,
                             int m) override{
                std::vector<real> result(n);
                for (int i = 0; i < n; i++) {
                    real sum = theta[m];
                    for (int j = 0; j < m; j++) {
                        sum += (X[i * m + j] * theta[j]);
                    }
                    result[i] = util::sigmoid(sum);
                }
                return result;
            }

            std::vector<int> predict(std::vector<real>& X,
                        int n,
                        int m) override {
                std::vector<real> probs=get_pred_prob(X, n, m);
                std::vector<int> result(n);
                for (int i = 0; i < n; i++) {
                    result[i] = (probs[i] > 0.5);
                }
                return result;
            }
        };
    }
}

