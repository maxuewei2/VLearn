//
// Created by xuewei on 2019/9/19.
//

#ifndef VLEARN_CLASSIFIER_H
#define VLEARN_CLASSIFIER_H


#include "../util/utils.h"
#include <string>
namespace vlearn {
    namespace model {
        class Classifier {
        protected:
            std::string name="Classifier";
        public:
            void f(){};
            virtual void fit(std::vector<real>& X0,
                            std::vector<int>& y,
                            int n,
                            int m) = 0;

            virtual std::vector<int> predict(std::vector<real>& X,
                                             int n,
                                             int m) = 0;

            virtual Classifier *__clone__() = 0;

            virtual void __delete__() = 0;

            //virtual ~Classifier()=0;

            virtual std::vector<real> get_pred_prob(std::vector<real>& X,
                                                   int n,
                                                   int m)=0;
        };
    }
}
#endif //MYPROJECT_CLASSIFIER_H
