import numpy as np
import pandas as pd

#导入数据集
#导入训练数据集
train = pd.read_csv('./train.csv')
#导入测试数据集
test = pd.read_csv('./test.csv')

#查看数据形状
print('训练数据集大小：',train.shape,'测试数据集大小：',test.shape)
#运行结果：训练数据集大小： (891, 12) 测试数据集大小： (418, 11)
'''
    数据描述。
    训练数据：共有891行，12列。
    测试数据：共有418行，11列。
'''

#合并数据集用来对两个数据集进行数据清洗
full = train.append(test,ignore_index=True)

#数据预处理
#Age、Embarked和Cabin均存在缺失数据，缺失数据需要进行数据清洗处理；
#缺失值处理常用方法：如果是数值类型，用平均值取代；如果是分类数据，用最常见的类别取代；使用模型预测缺失值，例如：K-NN。
#对于Age（年龄）和Fare（船票价格）两个数值类型字段，缺失的部分采用最简单的方法平均数来填充；
print('处理前')
full.info()
#年龄
full['Age'] = full['Age'].fillna(full['Age'].mean())
#船票价格
full['Fare'] = full['Fare'].fillna(full['Fare'].mean())
print('处理后')
full.info()

#检查数据处理是否正常
full.head()

#Embarked（登船港口）和Cabin（船舱号）属于分类数据，无法像数值类型数据采用均值填充；
#1）登船港口(Embarked）里面数据总数是1307，只缺失了2条数据，缺失比较少；
#2）船舱号(Cabin)里面数据总数是295，缺失了1309-295=1014，缺失率=1014/1309=77.5%，缺失比较大；
# 由于Embarked（登船港口）选择S的最多，为914个。因此，缺失值用S填充；
full['Embarked'] = full['Embarked'].fillna('s');
#Cabin缺失值数量较多，缺失值填充为U，表示未知（Uknow）;
full['Cabin'] = full['Cabin'].fillna('U');
#检查数据处理是否正常
full.head()
#查看最终缺少值处理情况
full.info()

#特征提取
#查看数据类型，分为3种数据类型。
'''
1.数值类型：乘客编号（PassengerId），年龄（Age），船票价格（Fare），同代直系亲属人数（SibSp），不同代直系亲属人数（Parch）
2.时间序列：无
3.分类数据：
1）有直接类别的
乘客性别（Sex）：男性male，女性female
登船港口（Embarked）：出发地点S=英国南安普顿Southampton，途径地点1：C=法国 瑟堡市Cherbourg，出发地点2：Q=爱尔兰 昆士敦Queenstown
客舱等级（Pclass）：1=1等舱，2=2等舱，3=3等舱
2）字符串类型：可能从这里面提取出特征来，也归到分类数据中
乘客姓名（Name）
客舱号（Cabin）
船票编号（Ticket）
'''


#对直接类别数据进行分类
#将性别的值映射为数值，男性（male）对应1，女性（female）对应0
sex_mapDict = {"male":1,"female":0}

#d对Series每个数据应用自定义的函数计算
full['Sex'] = full['Sex'].map(sex_mapDict)
#full.head()



#登船港口（Embarked）数据处理
#存放提取后的特征
embarkeDf = pd.DataFrame()

#使用get_dummies进行one-hot编码，产生虚拟变量（dummy variables），列名前缀是Embarked
embarkeDf = pd.get_dummies(full['Embarked'],prefix='Embarked')
#embarkeDf.head()

#添加one-hot编码产生的虚拟变量（dummy variables）到Tatanic数据集full
full = pd.concat([full,embarkeDf],axis=1)

#删除原来的登船港口（Embarked）
full.drop('Embarked',axis=1,inplace=True)
#full.head()

#客舱等级（Pcalss）数据处理
#存放提取后的特征
pclassDf = pd.DataFrame()

#使用get_dummies进行one-hot编码，列名前缀是Pclass
pclassDf = pd.get_dummies(full['Pclass'],prefix='Pclass')
#pclassDf.head()

#添加one-hot编码产生的虚拟变量（dummy variables）到Tatanic数据集full
full = pd.concat([full,pclassDf],axis=1)

#删除原来的客舱等级（Pcalss）
full.drop('Pclass',axis=1,inplace=True)
#full.head()



#字符串类型处理
#这里数据有：乘客姓名（Name）、客舱号（Cabin）、船票编号（Ticket），乘客姓名删除。


#存放客舱号（Cabin）信息
cabinDf = pd.DataFrame()
full['Cabin'] = full['Cabin'].map(lambda  c:c[0])

#使用get_dummies进行one-hot编码,列名前缀Cabin
cabinDf = pd.get_dummies(full['Cabin'],prefix='Cabin')
#cabinDf.head()

#添加one-hot编码产生的虚拟变量（dummy variables）到Tatanic数据集full
full = pd.concat([full,cabinDf],axis=1)

#删除原来的客舱号（Cabin））
full.drop('Cabin',axis=1,inplace=True)
#full.head()



#建立家庭人数和家庭类别
#存放家庭信息（Parch，SibSp）
familyDf = pd.DataFrame()
'''
家庭人数=同代直系亲属数（Parch）+不同代直系亲属数（SibSp）+乘客自己
'''
familyDf['FamilySize'] = full['Parch']+full['SibSp']+1
'''
家庭类别：
在此定义为：
小家庭Family_Single：家庭人数=1
中等家庭Family_Small: 2<=家庭人数<=4
大家庭Family_Large:家庭人数>4
'''
familyDf['Family_Single']=familyDf['FamilySize'].map(lambda s:1 if s==1 else 0)
familyDf['Family_Small']=familyDf['FamilySize'].map(lambda s:1 if 2<=s<=4 else 0)
familyDf['Family_Large']=familyDf['FamilySize'].map(lambda s:1 if 5<=s else 0)

#familyDf.head()

#添加one-hot编码产生的虚拟变量（dummy variables）到Tatanic数据集full
full = pd.concat([full,familyDf],axis=1)
#删除家庭信息（Parch，SibSp）

full.drop('Parch',axis=1,inplace=True)
full.drop('SibSp',axis=1,inplace=True)


#通过以上对Sex（性别）、Embarked（登船港口）、Pclass（客舱等级）和家庭人数（通过Parch、Sibsp计算得到）进行的特征处理操作，full特征总量如下：
full.shape



#相关性矩阵计算，计算各个特征的相关系数
corrDf = full.corr()
corrDf

#查看各个特征与生成情况（Survived）的相关系数，并按降序排列：
corrDf['Survived'].sort_values(ascending=False)



#根据各个特征与生成情况（Survived）的相关系数大小，我们选择了这几个特征作为模型的输入：客舱等级（pclassDf）、家庭大小（familyDf）、船票价格（Fare）、船舱号（cabinDf）、登船港口（embarkedDf）、性别（Sex）
#特征选择
full_X = pd.concat(
    [familyDf,
     full['Fare'],
     cabinDf,
     embarkeDf,
     full['Sex'],
     ],axis=1)


#建立模型

#获取原始训练数据集和预测数据集
#原始数据集有891行
sourceRow=891
#原始数据集：特征
source_X=full_X.loc[0:sourceRow-1,:]
#原始数据集：标签
source_y=full.loc[0:sourceRow-1,'Survived']
#预测数据集：特征
pred_X=full_X.loc[sourceRow:,:]
print(source_X.shape[0])
print(pred_X.shape[0])
#从原始数据集（source）中拆分出训练数据集（用于模型训练train），测试数据集（用于模型评估test）
#train_test_split是交叉验证中常用的函数，功能是从样本中随机的按比例选取train data和test data
#train_data：所要划分的样本特征集
#train_target：所要划分的样本结果
#test_size：样本占比，如果是整数的话就是样本的数量
from sklearn.model_selection import train_test_split
#建立模型用的训练数据集和预测数据集
train_X,test_X,train_y,test_y = train_test_split(source_X,source_y,train_size=.8)
#输出数据集大小
print(source_X.shape,train_X.shape,test_X.shape)
print(source_y.shape,train_y.shape,test_y.shape)




#使用朴素贝叶斯算法处理


