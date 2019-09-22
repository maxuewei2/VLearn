//
// Created by xuewei on 2019/9/21.
//

#ifndef VLEARN_SCALER_H
#define VLEARN_SCALER_H

#include "../util/utils.h"
namespace vlearn{
    namespace model{
        class Scaler{
        public:
            virtual void fit(std::vector<real>& X,int n,int m)=0;
            virtual void transform(std::vector<real>& X,int n,int m)=0;
        };
        class MinMaxScaler:public Scaler{
            std::vector<real> mins;
            std::vector<real> maxs;
        public:
            void fit(std::vector<real>& X,int n,int m)override {
                mins.resize(m);
                maxs.resize(m);
                real tmp,min,max;
                for(int i=0;i<m;i++){
                    min=X[i];
                    max=X[i];
                    for(int j=0;j<n;j++){
                        tmp=X[j*m+i];
                        min=min<tmp?min:tmp;
                        max=max>tmp?max:tmp;
                    }
                    mins[i]=min;
                    maxs[i]=max;
                }
            }
            void transform(std::vector<real>& X,int n,int m)override {
                real min,max,tmp;
                for(int i=0;i<m;i++){
                    min=mins[i];
                    max=maxs[i];
                    tmp=max-min;
                    for(int j=0;j<n;j++){
                        X[j*m+i]=(X[j*m+i]-min)/(tmp+0.0001)*2-1;
                    }
                }
            }
        };
        class StandardScaler:public Scaler{
        public:
            std::vector<real> means;
            std::vector<real> svars;
            void fit(std::vector<real>& X,int n,int m)override {
                means.resize(m);
                svars.resize(m);
                real tmp,mean,sum=0,sumv=0;
                for(int i=0;i<m;i++){
                    sum=0;
                    for(int j=0;j<n;j++){
                        sum+=X[j*m+i];
                    }
                    mean=sum/(real)n;
                    means[i]=mean;
                    sumv=0;
                    for(int j=0;j<n;j++){
                        tmp=X[j*m+i]-mean;
                        sumv+=(tmp*tmp);
                    }
                    sumv/=(real)n;
                    svars[i]=sqrt(sumv);
                }
            }
            void transform(std::vector<real>& X,int n,int m)override {
                real mean,svar,tmp;
                for(int i=0;i<m;i++){
                    mean=means[i];
                    svar=svars[i];
                    for(int j=0;j<n;j++){
                        X[j*m+i]=(X[j*m+i]-mean)/(svar+0.000001);
                    }
                }
            }
        };
    }
}
#endif //VLEARN_SCALER_H
