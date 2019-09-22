//
// Created by xuewei on 2019/9/19.
//

#ifndef VLEARN_ONEVSREST_H
#define VLEARN_ONEVSREST_H

#include "classifier.h"
#include <map>

namespace vlearn {
    namespace model {
        class OneVsRest : public vlearn::model::Classifier {
            std::vector<Classifier *> clfs;
            Classifier &given_clf;
            std::vector<int> ys;
            int y_num = 0;
            std::vector<real> pred_probs;
        public:
            explicit OneVsRest(Classifier &givenClf) : given_clf(givenClf) {
            }

            void fit(std::vector<real> &X,
                     std::vector<int> &y,
                     int n,
                     int m) override {
                for (auto &it:clfs) {
                    delete it;
                }
                std::map<int, int> y_map;
                for (int i = 0; i < n; i++) { y_map[y[i]] = 1; }
                y_num = y_map.size();
                clfs.resize(y_num);
                ys.resize(y_num);
                std::vector<int> y_cp(n);
                int clf_i = 0;
                for (auto &it : y_map) {
                    int target_y = it.first;
                    ys[clf_i] = target_y;
                    for (int j = 0; j < n; j++) {
                        y_cp[j] = y[j] == target_y ? 1 : 0;
                    }
                    clfs[clf_i] = given_clf.__clone__();
                    clfs[clf_i]->fit(X, y_cp, n, m);
                    clf_i++;
                }
            }

            std::vector<real> get_pred_prob(std::vector<real> &X,
                                             int n,
                                             int m) override {
                pred_probs.resize(n * y_num);
                for (int j = 0; j < y_num; j++) {
                    std::vector<real> one_prob = clfs[j]->get_pred_prob(X, n, m);
                    for (int i = 0; i < n; i++) {
                        pred_probs[i * y_num + j] = one_prob[i];
                    }
                }
                return pred_probs;
            }

            std::vector<int> predict(std::vector<real>& X,
                                     int n,
                                     int m) override {
                get_pred_prob(X,n,m);
                std::vector<int> result(n);
                for(int i=0;i<n;i++){
                    int max_index=-1;
                    real max_prob=-1;
                    for(int j=0;j<y_num;j++){
                        real tmp=pred_probs[i*y_num+j];
                        if(tmp>max_prob){
                            max_prob=tmp;
                            max_index=j;
                        }
                    }
                    result[i]=ys[max_index];
                }
                return result;
            }

            OneVsRest *__clone__() override {
                return nullptr;
            }

            ~OneVsRest() {
                this->__delete__();
            }

            void __delete__() override {
                for (auto &it:clfs) {
                    it->__delete__();
                }
            }
        };
    }
}
#endif //MYPROJECT_ONEVSREST_H
