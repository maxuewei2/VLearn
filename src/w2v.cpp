#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <map>
#include <vector>
#include <algorithm>
#include <time.h>
#include <unordered_map>
#include <stdio.h>
#include <bitset>

#define MAX_STRING  500
#define SIGMOID_BOUND 6
#define NEG_SAMPLING_POWER 0.75
#define BITSIZE 100
#define LINE_LEN 10000


const int hash_table_size = 30000000;
const int neg_table_size = 1e8;
const int sigmoid_table_size = 1000;

typedef float real;
struct Node{
int deg=0;
int id=-1;
bool visited=0;
real* emb=NULL;
Node* left=NULL;
Node* right=NULL;
};
typedef struct Node Node;

char train_file[MAX_STRING], embedding_file[MAX_STRING];
int dim=128,iter=5,num_threads=4,window_size=10;
real learning_rate_0=0.05;
real learning_rate=learning_rate_0;
int min_count=5;
real sample=1e-4;

std::unordered_map<std::string, int> id_map;
std::string* rid_map;
std::vector<int> degree_map;
real ** embs;
int num_words;
int num_corpus;
std::bitset<BITSIZE>* code_map;
char* code_len;
Node* hs_root;

char time_buffer[26];

char* get_time(){
    time_t timer;
    struct tm* tm_info;
    time(&timer);
    tm_info = localtime(&timer);
    strftime(time_buffer, 26, "%Y-%m-%d %H:%M:%S", tm_info);
    return time_buffer;
}
int preprocess_corpus(){
    std::cout<<get_time()<<"  preprocessing file "<<train_file<<" ..."<<std::endl;
    std::ifstream ifs;
    ifs.open(train_file, std::ifstream::in);
    std::string word;
    int id=0;
    while (ifs>> word){
        num_corpus++;
        if(num_corpus%1000==0){
            printf("%dk words\r",num_corpus/1000);
            fflush(stdout);
        }
        if(id_map.find(word)==id_map.end()){
            id_map[word]=id;
            degree_map.push_back(0);
            id++;
        }
        degree_map[id_map[word]]++;
    }
    num_words=id;
    std::cout<<"      vocab size:  "<<num_words<<"\n      corpus size: "<<num_corpus<<std::endl;
    rid_map=new std::string[num_words];
    int i=0;
    for (std::unordered_map<std::string,int>::iterator it=id_map.begin(); it!=id_map.end(); ++it,++i){
        rid_map[it->second]=(it->first);
    }
    //~ for (std::map<std::string,int>::iterator it=id_map.begin(); it!=id_map.end(); ++it)
    //~ std::cout << it->first << " => " << it->second << '\n';
    
    //~ std::cout<<std::endl<<std::endl;
    
    //~ for (std::map<int,int>::iterator it=degree_map.begin(); it!=degree_map.end(); ++it)
    //~ std::cout << it->first << " => " << it->second << '\n';
    
    //~ std::cout<<std::endl<<std::endl;
    
    //~ for (std::map<int,std::string>::iterator it=rid_map.begin(); it!=rid_map.end(); ++it)
    //~ std::cout << it->first << " => " << it->second << '\n';
    
    ifs.close();
    return 0;
}
int process_2(){
    std::cout<<get_time()<<"  process 2..."<<std::endl;
    int *ids=new int[num_words];
    for(int i=0;i<num_words;i++)ids[i]=i;
    std::sort(ids,ids+num_words,[](int &a,int &b){return degree_map[a]<degree_map[b];});
    int start=0;
    while(degree_map[ids[start]]<min_count){start++;}
    int degree_sum=0;
    for(int i=start;i<num_words;i++){degree_sum+=degree_map[ids[i]];}
    int *newid_map=new int[num_words];
    for(int i=0;i<num_words;i++)newid_map[i]=-1;
    int nid=0;
    for(int i=start;i<num_words;i++){
        int id=ids[i];
        real x =rand()/(real)RAND_MAX;
        //std::cout<<x<<std::endl;
        real t=sample/(degree_map[id]/(real)degree_sum);
        if(1-sqrt(t)-t>x){std::cout<<"xxxxxxx"<<std::endl;continue;}
        newid_map[id]=nid++;
    }
    num_words=0;
    num_corpus=0;
    std::vector<int> degree_map_bak(degree_map);
    for (std::unordered_map<std::string,int>::iterator it=id_map.begin(); it!=id_map.end(); ++it){
        int oid=it->second;
        nid=newid_map[oid];
        id_map[it->first]=nid;
        if(nid!=-1){rid_map[nid]=(it->first);num_words++;degree_map[nid]=degree_map_bak[oid];num_corpus+=degree_map[nid];}
    }
    std::cout<<"      vocab size:  "<<num_words<<"\n      corpus size: "<<num_corpus<<std::endl;
    delete ids;
    delete newid_map;
    return 0;
}
int init_vec(real* vec,int size, bool zero_flag=0){
    if(zero_flag){
        for(int j=0;j<size;j++){
            vec[j]=0.0;
        }    
        return 0;
    }
    for(int j=0;j<size;j++){
        vec[j]=(rand() / (real)RAND_MAX - 0.5) / size;
    }
    return 0;
}
int initialize_embedding(){
    std::cout<<get_time()<<"  initializing embedding..."<<std::endl;
    embs=new real*[num_words];
    for(int i=0;i<num_words;i++){
        embs[i]=new real[dim];
        init_vec(embs[i],dim,0);
    }
    return 0;
}

int get_min_two(Node* nodes,int i,std::vector<Node*>& node_queue,int j,Node *&ca, Node *&cb){
    Node *a=i<num_words?nodes+i:NULL,*b=(i+1)<num_words?nodes+i+1:NULL;
    Node *c=j<int(node_queue.size())?node_queue[j]:NULL,*d=j+1<int(node_queue.size())?node_queue[j+1]:NULL;
    std::vector<Node*> tmp;
    if(a!=NULL)tmp.push_back(a);
    if(b!=NULL)tmp.push_back(b);
    if(c!=NULL)tmp.push_back(c);
    if(d!=NULL)tmp.push_back(d);
    if(tmp.size()<=1)return 0;
    if(tmp.size()>2)std::stable_sort(tmp.begin(),tmp.end(),[](Node* ta,Node*tb){return ta->deg<tb->deg;});
    ca=tmp[0];
    cb=tmp[1];
    ca->visited=1;
    cb->visited=1;
    //std::cout<<ca->id<<" id "<<cb->id<<std::endl;
    return 1;
}
int get_code(Node* root,int level,std::bitset<BITSIZE>& bittmp,bool x){
    //std::cout<<"level "<<level<<std::endl;
    if(root==NULL)return 1;
    if(level!=-1){
        bittmp[level]=x;
    }
    int nlevel=level+1;
    std::bitset<BITSIZE> tmp=bittmp;
    int lflag=get_code(root->left,nlevel,bittmp,0);
    bittmp=tmp;
    int rflag=get_code(root->right,nlevel,bittmp,1);
    if(lflag&rflag){
        //bittmp[level+1]='\0';
        int id=root->id;
        //code_map[id]=new char[level+2];
        code_map[id]=tmp;
        code_len[id]=level+1;
    }
    //~ if(root->id==-1){delete root;}
    return 0;
}
int create_tree(){
    std::cout<<get_time()<<"  creating tree..."<<std::endl;
    Node *nodes=new Node[num_words];
    for(int i=0;i<num_words;i++){
        nodes[i].id=i;
        nodes[i].deg=degree_map[i];
        //real *emb_tmp=new real[dim];
        //init_vec(emb_tmp,dim,1);
        //nodes[i].emb=emb_tmp;
    }
    std::sort(nodes,nodes+num_words,[](Node& a,Node& b){return a.deg<b.deg;});
    //~ for(int i=0;i<num_words;i++){
        //~ std::cout<<nodes[i].id<<" "<<nodes[i].deg<<" "<<std::endl;
    //~ }
    int i=0;
    int j=0;
    Node *a, *b;
    std::vector<Node*> node_queue;
    while(get_min_two(nodes,i,node_queue,j,a,b)){
        //~ std::cout<<"deg "<<a->deg<<" "<<b->deg<<" "<<a->id<<" "<<b->id<<std::endl;
        Node* father=new Node();
        father->deg=a->deg+b->deg;
        father->left=a;
        father->right=b;
        real *emb_tmp=new real[dim];
        init_vec(emb_tmp,dim,1);
        father->emb=emb_tmp;
        node_queue.push_back(father);
        while(i<num_words&&nodes[i].visited==1){i++;}
        while(j<int(node_queue.size())&&node_queue[j]->visited==1)j++;
    }
    hs_root=node_queue[j];
    std::bitset<BITSIZE> bittmp;
    code_map=new std::bitset<BITSIZE>[num_words];
    code_len=new char[num_words];
    std::cout<<get_time()<<"  getting code..."<<std::endl;
    get_code(hs_root,-1,bittmp,0);
    
    //~ std::ofstream ofs;
    //~ ofs.open("code", std::ofstream::out);
    //~ for(int i=0;i<num_words;i++){
        //~ ofs<<nodes[i].id<<" "<<nodes[i].deg<<" "<<code_map[nodes[i].id]<<" "<<int(code_len[nodes[i].id])<<std::endl;
    //~ }
    return 0;
}
real sigmoid(real x){
    return 1/(1+exp(-x));
}
int add_vec(real *a,real *b,real *c,int m){
    for(int i=0;i<m;i++){
        c[i]=a[i]+b[i];
    }
    return 0;
}
int multi_vec(real* a,real x,real* b,int m){
    for(int i=0;i<m;i++){
        b[i]=a[i]*x;
    }
    return 0;
}
real inner_product(real *a,real *b,int m){
    real x=0.0;
    for(int i=0;i<m;i++){
        x+=(a[i]*b[i]);
    }
    return x;
}
int one_pair(std::string* line,int n_words,int anchor){
    int id=id_map[line[anchor]];
    if(id==-1)return 0;
    real* x=new real[dim];
    init_vec(x,dim,1);
    real** context=new real*[window_size*2];
    int j=0;
    for(int i=-window_size;i<=window_size;i++){
        //std::cout<<line[anchor+i]<<" ";
        if(i==0)continue;
        int cid=id_map[line[anchor+i]];
        if(cid==-1)continue;
        real *cv=embs[cid];
        add_vec(x,cv,x,dim);
        context[j++]=cv;
    }
    //std::cout<<std::endl;
    if(j==0)return 0;
    multi_vec(x,1.0/(real)j,x,dim);//????????????似乎很重要,加了之后of等常用词就下去了
    Node* p=hs_root;
    std::bitset<BITSIZE>& code=code_map[id];
    real* e=new real[dim];
    init_vec(e,dim,1);
    real* mt=new real[dim];
    for(int i=0;i<code_len[id];i++){
        //std::cout<<code[i];
        real * theta=p->emb;
        real inp=inner_product(x,theta,dim);
        if(inp<=-6 || inp>=6)continue;//?????????????似乎加不加没区别 但为什么没区别
        real sig=sigmoid(inp);
        real l=code[i]?1.0:0.0;
        real g=(1.0-l-sig)*learning_rate;
        multi_vec(theta,g,mt,dim);
        add_vec(e,mt,e,dim);
        multi_vec(x,g,mt,dim);
        add_vec(theta,mt,theta,dim);
        p=code[i]?p->right:p->left;
    }
    //std::cout<<line[anchor]<<std::endl;
    //multi_vec(e,1/j,e,dim);//?????????????
    for(int i=0;i<j;i++){
        add_vec(context[i],e,context[i],dim);
    }
    delete x;
    delete e;
    delete mt;
    delete context;
    return 1;
}
int one_line(std::string* line, int n_words){
    std::string anchor;
    int c=0;
    for(int i=window_size;i<n_words-window_size;i++){
        int tmp=window_size;
        window_size=rand()%window_size;//?????????????????似乎影响很大?
        if(!window_size){window_size=tmp;continue;}
        c+=one_pair(line,n_words,i);
        window_size=tmp;
    }
    return c;
}

int one_iter(int iter_i){
    std::cout<<get_time()<<"  iter "<<iter_i+1<<std::endl;
    std::ifstream ifs;
    ifs.open(train_file, std::ifstream::in);
    std::string line[LINE_LEN];
    int count=0;
    while(ifs.good()){
        int n_words=0;
        while (ifs>> line[n_words++]){if(n_words>=LINE_LEN)break;}
        count+=one_line(line,n_words);
        
        int words_done=iter_i*num_corpus+count;
        learning_rate=(1.0-(words_done*1.0)/(num_corpus+1.0))*learning_rate_0;
        if(learning_rate<10e-4*learning_rate_0){
            learning_rate=10e-4*learning_rate_0;
        }
        std::cout<<"\r"<<learning_rate;
        fflush(stdout);
    }
    return 0;
}

int word2vec(){
    std::cout<<get_time()<<"  begin word2vec"<<std::endl;
    for(int i=0;i<iter;i++){
        one_iter(i);
    }
    return 0;
}

int save_emb(){
    std::cout<<get_time()<<"  saving embeddings..."<<std::endl;
    std::ofstream ofs;
    ofs.open(embedding_file, std::ofstream::out);
    ofs<<num_words<<" "<<dim<<std::endl;
    for(int i=0;i<num_words;i++){
        ofs<<rid_map[i];
        real* emb=embs[i];
        for(int j=0;j<dim;j++){
            ofs<<" "<<emb[j];
        }
        ofs<<std::endl;    
    }
    ofs.close();
    return 0;
}

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

int main(int argc,char** argv)
{
    int i;
    if (argc == 1) {
        printf("\nword2vec\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-input <file>\n");
        printf("\t\tUse corpus data from <file> to train the model\n");
        printf("\t-output <file>\n");
        printf("\t\tUse <file> to save the learnt embeddings\n");
        printf("\t-dimensions <int>\n");
        printf("\t\tSet dimension of vertex embeddings; default is 100\n");
        printf("\t-window-size <int>\n");
        printf("\t\tSet window size; default is 10\n");
        printf("\t-min-count <int>\n");
        printf("\t\tSet min count; default is 5\n");
        printf("\t-iter <int>\n");
        printf("\t\tSet iter num; default is 5\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 1)\n");
        printf("\nExamples:\n");
        printf("./word2vec -input test.txt -output vec.txt -dimensions 128 -num-walks 20 -walk-length 80 -window-size 10 -threads 20\n\n");
        return 0;
    }
    if ((i = ArgPos((char *)"-input", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(embedding_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-dimensions", argc, argv)) > 0) dim = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-window-size", argc, argv)) > 0) window_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
    preprocess_corpus();
    process_2();
    initialize_embedding();
    create_tree();
    word2vec();
    save_emb();
    return 0;
}
