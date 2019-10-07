#include "LogisticReg.h"
double ObjFunc(const MatrixXd & X,const VectorXd &y,const VectorXd &w)
{
    return ((w.transpose()*X).array().exp()+1).log().sum()-w.transpose()*X*y;
};
VectorXd LogisticReg( const MatrixXd & Data,const VectorXd &y)

{
    //maxiter 最大迭代次数
    int maxiter=2000;
    VectorXd loss(maxiter+1,1);
    loss=VectorXd::Zero(maxiter+1,1);

    int n=Data.cols();
    int d=Data.rows();

    //X: 在Data基础上加全一行
    MatrixXd X(d+1,n);
    X.block(0,0,d,n)=Data.block(0,0,d,n);
    X.block(d,0,1,n)=MatrixXd::Ones(1,n);

    VectorXd w(d+1,1);
    w=VectorXd::Random(d+1,1);

    VectorXd gradient(d+1,1);
    gradient=VectorXd::Zero(d+1,1);
    loss(0)=ObjFunc(X,y,w);
    double prevloss=loss(0);
    //learn_rate_init 初始学习率
    double learn_rate_init=1e8;
    double learn_rate=learn_rate_init;
    // alpha beta：Backtracking line search的参数
    double alpha=0.5;
    double beta=0.5;
    //abstol 梯度阈值,当梯度小于abstol，停止迭代
    double abstol=0.001;
    gradient=X*((((X.transpose()*w).array().exp())/ ((X.transpose()*w).array().exp()+1)).matrix()-y);
    int iter=1;
    for(;iter<=maxiter; iter++)
    {

        learn_rate=learn_rate_init;
        gradient=X*((((X.transpose()*w).array().exp())/ ((X.transpose()*w).array().exp()+1)).matrix()-y);
        if(pow(gradient.norm(),2)<abstol)
            break;

        //Backtracking line search
        while(ObjFunc(X,y,w-learn_rate*gradient)>(prevloss-alpha*learn_rate*pow(gradient.norm(),2)))
        {
            learn_rate=beta*learn_rate;
        }

        w=w-learn_rate*gradient;
        loss(iter)=ObjFunc(X,y,w);
        prevloss=loss(iter);
        cout<<"Iter "<<iter<<": learning rate= "<<learn_rate<<" ,loss= "<<loss(iter)<<endl;
    }
    return w;
};
VectorXd LogisticRegPredict(const MatrixXd & Data,const VectorXd &w)
{
    int n=Data.cols();
    int d=Data.rows();
    //X: 在Data基础上加全一行
    MatrixXd X(d+1,n);
    X.block(0,0,d,n)=Data.block(0,0,d,n);
    X.block(d,0,1,n)=MatrixXd::Ones(1,n);

    ArrayXXd result(n,1);
    result=1/(1+((-X.transpose()*w).array().exp()));
    for(int i=0; i<n; i++)
    {
        if(result(i)<0.5)
            result(i)=0;
        else
            result(i)=1;
    }
    return result.matrix();
};


