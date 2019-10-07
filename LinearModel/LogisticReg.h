#include<Eigen/Dense>
#include<iostream>
#include <cmath>
using namespace Eigen;
using namespace std;
double ObjFunc(const MatrixXd & X,const VectorXd &y,const VectorXd &w);
/*�����߼��ع�Ŀ�꺯��ֵ
Input:
X �������� ά��Ϊd+1 * n dΪ����ά�ȣ�nΪ������
y �������ݵ���ʵ��ǩ ά����n*1
w ����Ȩ��,����ƫ����,ά����d+1*1
Output:
//�߼��ع�Ŀ�꺯��ֵ;
*/
VectorXd LogisticReg( const MatrixXd & Data,const VectorXd &y);
/*ѵ�����ݵõ�����Ȩ��
Input:
X �������� ά��Ϊd*n dΪ����ά�ȣ�nΪ������
y �������ݵ���ʵ��ǩ ά����n*1
Output:
//w ����Ȩ��,����ƫ����,ά����d+1*1;
*/
VectorXd LogisticRegPredict(const MatrixXd & Data,const VectorXd &w);
/*Ԥ�����ݱ�ǩ
Input:
X �������� ά��Ϊd*n dΪ����ά�ȣ�nΪ������
w ����Ȩ��,����ƫ���� ά����d+1*1;
Output:
 VectorXdԤ����,ά����n*1
*/

