#include<Eigen/Dense>
#include<iostream>
#include <cmath>
using namespace Eigen;
using namespace std;
double ObjFunc(const MatrixXd & X,const VectorXd &y,const VectorXd &w);
/*计算逻辑回归目标函数值
Input:
X 输入数据 维度为d+1 * n d为特征维度，n为样本数
y 输入数据的真实标签 维度是n*1
w 特征权重,包括偏移项,维度是d+1*1
Output:
//逻辑回归目标函数值;
*/
VectorXd LogisticReg( const MatrixXd & Data,const VectorXd &y);
/*训练数据得到特征权重
Input:
X 输入数据 维度为d*n d为特征维度，n为样本数
y 输入数据的真实标签 维度是n*1
Output:
//w 特征权重,包括偏移项,维度是d+1*1;
*/
VectorXd LogisticRegPredict(const MatrixXd & Data,const VectorXd &w);
/*预测数据标签
Input:
X 输入数据 维度为d*n d为特征维度，n为样本数
w 特征权重,包括偏移项 维度是d+1*1;
Output:
 VectorXd预测结果,维度是n*1
*/

