//
// Created by xuewei on 2019/9/21.
//
#include "vlearn.h"
#include <cstdio>

#define MAX_STR 1000

int test_emb() {
    char* tmpc=new char[MAX_STR];
    printf("read X\n");
    FILE* f=fopen("../data/gw.emb", "r");
    int n,m;
    fscanf(f,"%d %d", &n,&m);
    printf("%d %d\n",n,m);
    std::vector<real> X(n*m);
    std::map<std::string,int> words;
    for(int i=0;i<n;i++){
        fscanf(f,"%s",tmpc);
        std::string tmp(tmpc);

        words[tmp]=i;
        for(int j=0;j<m;j++){
            fscanf(f,"%f",&X[i*m+j]);
        }
        if(i<10){
            printf("%s",tmpc);
            for(int j=0;j<m;j++) {
                printf(" %.5f",X[i*m+j]);
            }
            printf("\n");
        }
    }
    fclose(f);
    printf("read y\n");
    FILE* fy=fopen("../data/usa-airports.group", "r");
    std::vector<int> y(n);
    for(int i=0;i<n;i++){
        fscanf(fy,"%s",tmpc);
        std::string tmp(tmpc);
        int l;
        fscanf(fy,"%d",&l);
        y[words[tmp]]=l;
    }
    fclose(fy);
    for (int k = 0; k < 10; ++k) {
        printf("%d ",y[k]);
    }
    printf("\n");
    vlearn::model::OneVsRest ovr(*new vlearn::model::LR(1000000,0.001));
    printf("fit\n");
    float train_ratio=0.8;
    int train_num=n*train_ratio;
    int test_num=n-train_num;
    std::vector<real> trainX=std::vector<real>(X.begin(),X.begin()+train_num * m);
    std::vector<int> trainy=std::vector<int>(y.begin(),y.begin()+train_num);
    std::vector<real> testX=std::vector<real>(X.begin()+train_num*m,X.end());
    std::vector<int> testy=std::vector<int>(y.begin()+train_num,y.end());

    vlearn::model::Scaler* scaler=new vlearn::model::MinMaxScaler();
    //vlearn::model::Scaler* scaler=new vlearn::model::StandardScaler();
    scaler->fit(trainX,train_num,m);
    scaler->transform(trainX,train_num,m);
    scaler->transform(testX,test_num,m);

    ovr.fit(trainX,trainy,train_num,m);
    printf("done fitting\n");

    std::vector<int> result=ovr.predict(testX,test_num,m);
    printf("acc: %f\n",vlearn::util::accuracy(testy,result));
//    for (int i = 0; i < m + 1; i++) {
//        printf("%.4f ", ovr.get_coef_()[i]);
//    }
    return 0;
}

int test_b(){
    printf("reading data\n");
    FILE *f = fopen("../data/breast-cancer-wisconsin.data1", "r");
    if (f == nullptr)printf("file reading error.\n");
    int n = 699, m = 9, id;
    printf("%d %d\n", n, m);
    std::vector<real> X(n * m);
    std::vector<int> y(n);
    for (int i = 0; i < n; i++) {
        fscanf(f, "%d", &id);
        for (int j = 0; j < m; j++) {
            fscanf(f, ",%f", &X[i * m + j]);
        }
        int tmp;
        fscanf(f, ",%d", &tmp);
        y[i] = tmp == 2 ? 1 : 0;
        if (i < 10) {
            printf("%d ", id);
            for (int j = 0; j < m; j++) {
                printf("%.2f ", X[i * m + j]);
            }
            printf("%d %d\n", y[i], tmp);
        }
    }
    fclose(f);

    vlearn::model::LR lr(1000000, 0.001);
    printf("fitting\n");
    float train_ratio = 0.8;
    int train_num = n * train_ratio;
    int test_num = n - train_num;
    std::vector<real> trainX=std::vector<real>(X.begin(),X.begin()+train_num * m);
    std::vector<int> trainy=std::vector<int>(y.begin(),y.begin()+train_num);
    std::vector<real> testX=std::vector<real>(X.begin()+train_num * m,X.end());
    std::vector<int> testy=std::vector<int>(y.begin()+train_num,y.end());
    vlearn::model::Scaler* scaler=new vlearn::model::MinMaxScaler();
    scaler->fit(trainX,train_num,m);
    scaler->transform(trainX,train_num,m);
    scaler->transform(testX,test_num,m);

    lr.fit(trainX, trainy, train_num, m);
    printf("done fitting\n");
    std::vector<int> result = lr.predict(testX , test_num, m);
    printf("acc: %f\n", vlearn::util::accuracy(testy, result));
//    for (int i = 0; i < m + 1; i++) {
//        printf("%.4f ", lr.theta[i]);
//    }
    return 0;
}
int main() {
    test_emb();
//    test_b();
    return 0;
}
