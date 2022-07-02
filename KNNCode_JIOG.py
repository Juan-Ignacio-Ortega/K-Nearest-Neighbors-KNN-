import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import random as rd
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/drive')

DB = pd.read_csv('/content/drive/MyDrive/ML_TSIV/KNN/TumorDataset.csv')

DB = DB[DB.columns[1:-1]]

DB.head()

sns.pairplot(DB, vars = DB.columns[1:6], hue = DB.columns[0])

Titles = list(DB.columns)

Titles_str = []
for title in Titles:
  if type(DB[title][0]) == str:
    Titles_str.append(title)

for title in Titles_str:
  types = pd.unique(DB[title])
  i = 0
  for data in types:
    data_loc = np.where(DB[title] == data)[0]
    for pos in data_loc:
      DB[title][pos] = i
    i += 1

DB.head()

NoAtributos = len(DB.T)
NoInstancias = len(DB)

DBar = DB.to_numpy()

MaximoDeAtributos = []
MinimoDeAtributos = []
for idx in range(NoAtributos):
  CaractMax = max(DBar.T[idx])
  CaractMin = min(DBar.T[idx])
  MaximoDeAtributos.append(CaractMax)
  MinimoDeAtributos.append(CaractMin)

Q1 = []
Q3 = []
for idx in range(NoAtributos):
  if str(type(DBar[0][idx]))[8 : -2] != 'str':
    atrib = DBar.T[idx].tolist()
    atrib.sort()

    NoCuartil1 = 0.25 * (NoInstancias + 1)
    if str(type(NoCuartil1))[8 : -2] != 'int':
      pos1 = round(NoCuartil1)
      if pos1 < NoCuartil1:
        pos2 = pos1 + 1
      else:
        pos2 = pos1 - 1
      NoCuartil1 = round((pos1 + pos2) / 2)
    Cuartil1 = atrib[NoCuartil1 + 1]
    incremento = 1
    while True:
      if str(Cuartil1) == 'nan':
          Cuartil1 = atrib[NoCuartil1]
          NoCuartil1 -= 1
      else:
        break

    NoCuartil3 = 0.7 * (NoInstancias + 1)
    if str(type(NoCuartil3))[8 : -2] != 'int':
      pos1 = round(NoCuartil3)
      if pos1 < NoCuartil3:
        pos2 = pos1 + 1
      else:
        pos2 = pos1 - 1
      NoCuartil3 = round((pos1 + pos2) / 2)
    Cuartil3 = atrib[NoCuartil3 + 1]
    while True:
      if str(Cuartil3) == 'nan':
          Cuartil3 = atrib[NoCuartil3]
          NoCuartil3 -= 1
      else:
        break

  Q1.append(Cuartil1)
  Q3.append(Cuartil3)
Q1 = np.array(Q1)
Q3 = np.array(Q3)

MaximoNormalizado = 1
MinimoNormalizado = 0
RangoNormalizado = MaximoNormalizado - MinimoNormalizado

DBNorm = []
for idx in range(NoAtributos):
  
  CaractNorm = []
  if str(type(DBar[0][idx]))[8 : -2] != 'str':
    
    RangodeDatos = MaximoDeAtributos[idx] - MinimoDeAtributos[idx]
    for idx2 in range(NoInstancias):

      if str(DBar[idx2][idx]) != 'nan':
        D = DBar[idx2][idx] - MinimoDeAtributos[idx]
        DPct = D / RangodeDatos
        dNorm = RangoNormalizado * DPct
        Normalizado = MinimoNormalizado + dNorm
        CaractNorm.append(Normalizado)
      else:
        CaractNorm.append(DBar[idx2][idx])
  
  else:
    for idx2 in range(NoInstancias):
      CaractNorm.append(DBar[idx2][idx])
  
  DBNorm.append(CaractNorm)

DBNar = np.array(DBNorm)

DBNarT = DBNar.T
showDB = pd.DataFrame(DBNarT)
showDB.head()

RDBNar = rd.sample(DBNar.T.tolist(), k=len(DBNar.T))
RDBNar = np.array(RDBNar).T

X = DBNar[1:]
X = X.T
Y = DBNar[0]

def MSE(MSEpred, MSEreal):
  MSEsize = len(MSEreal)
  MSEt = 0
  for MSEidx in range(MSEsize):
    MSEt += (MSEpred[MSEidx] - MSEreal[MSEidx])**2
  MSEf = MSEt / MSEsize
  return(MSEf)

def TC(TCpred, TCreal):
  TCsize = len(TCpred)
  TCt = 0
  for TCidx in range(TCsize):
    if TCpred[TCidx] != TCreal[TCidx]:
      TCt += 1
  TCf = 1 - TCt / TCsize
  return(TCf)

def MatConf(MCpred, MCreal):
  MCsize = len(MCpred)
  MCTP = 0
  MCFN = 0
  MCFP = 0
  MCTN = 0
  for MCidx in range(MCsize):
    if MCreal[MCidx] == 1:
      if MCpred[MCidx] == 1:
        MCTP += 1
      else:
        MCFN += 1
    else:
      if MCpred[MCidx] == 1:
        MCFP += 1
      else:
        MCTN += 1
  return(MCTP, MCFN, MCFP, MCTN)

def TdC(TdCTP, TdCFN, TdCFP, TdCTN):
  TdCExact = (TdCTP + TdCTN) / (TdCTP + TdCTN + TdCFP + TdCFN)
  TdCPre = TdCTP / (TdCTP + TdCFP)
  TdCSens = TdCTP / (TdCTP + TdCFN)
  TdCF1 = (2 * TdCTP) / (2 * TdCTP + TdCFP + TdCFN)
  return([TdCExact, TdCPre, TdCSens, TdCF1])

def Estadisticas(EYPred, EYreal):
  EMSE = MSE(EYPred, EYreal)
  ETC = TC(EYPred, EYreal)
  ETP, EFN, EFP, ETN = MatConf(EYPred, EYreal)
  EExact, EPrecis, ESens, EF1 = TdC(ETP, EFN, EFP, ETN)
  Estadistica = [EMSE, ETC, EExact, EPrecis, ESens, EF1]
  return(Estadistica)

class KNN():
  def __init__(self):
    self.bestK = 3

  def fit(self, fXTrainP, fYTrainP, fXTestP, fYTestP, fEpocasP = 4, fpriorityP = 'Precision'):
    self.fXtrain = fXTrainP
    self.fYtrain = fYTrainP
    self.fXtest = fXTestP
    self.fYtest = fYTestP
    self.fpriority = fpriorityP
    self.fEpocas = fEpocasP
    self.fKfin = 3 + 2 * (self.fEpocas - 1)
    self.bestK = self.SelectK()
  
  def PrednewPoint(self, KNNXnewPoint):
    self.newYpred = self.XpredP(KNNXnewPoint)
    return(self.newYpred)

  def Euc_Dist(self, EuX1, EuX2):
    EuInSqua = 0
    for EuIdx in range(len(EuX1)):
      EuInSqua += (EuX2[EuIdx] - EuX1[EuIdx])**2
    EucDistance = EuInSqua**0.5
    return(EucDistance)
  
  def XDistances(self, XNew):
    XDists = []
    for Xpoint in self.fXtrain:
      XDists.append(self.Euc_Dist(XNew, Xpoint))
    return(XDists)

  def KClosest(self, Knew):
    KDists = self.XDistances(Knew)
    KsortDists = sorted(KDists)
    KClosestDist =  KsortDists[:self.fKactual]
    KCposs = []
    for KCDist in KClosestDist:
      KCp = np.where(KDists == KCDist)
      KCposs.append(KCp[0][0])
    self.KClosestPoints = []
    for KCpos in KCposs:
      self.KClosestPoints.append(self.fXtrain[KCpos])
    return(self.KClosestPoints)

  def NumPointsxCat(self, Nnew):
    NClosestPoints = self.KClosest(Nnew)
    Npos = []
    for Npoint in NClosestPoints:
      NW = np.where(self.fXtrain == Npoint)
      Npos.append(NW[0][0])
    CatxPoint = []
    for Nelement in Npos:
      CatxPoint.append(self.fYtrain[Nelement])
    return(CatxPoint)

  def XpredP(self, XnewP):
    XCatP = self.NumPointsxCat(XnewP)
    XCounterP = Counter(XCatP)
    first = XCounterP.most_common(1)
    XYpredP = first[0][0]
    return(XYpredP)

  def PredAllP(self):
    PAYpred = []
    for PApoint in self.fXtest:
      PAYpred.append(self.XpredP(PApoint))
    return(PAYpred)

  def fstatistics(self):
    self.fKactual = 3
    fEstadisticas = []
    print(str(self.fpriority), 'con', str(self.fEpocas), 'K distintas:')
    while self.fKactual <= self.fKfin:
      fYpred = self.PredAllP()
      fEstadistica = Estadisticas(fYpred, self.fYtest)
      print(str(self.fpriority), 'con k =', str(self.fKactual) + ':', str(fEstadistica[3]) + '.')
      fEstadisticas.append(fEstadistica)
      self.fKactual += 2
    return(fEstadisticas)
  
  def SelectK(self):
    SEstadisticas = self.fstatistics()
    SMSE = []
    STC = []
    SExact = [] 
    SPrecis = []
    SSens = []
    SF1 = []
    for Selement in SEstadisticas:
      SMSE.append(Selement[0])
      STC.append(Selement[1])
      SExact.append(Selement[2])
      SPrecis.append(Selement[3])
      SSens.append(Selement[4])
      SF1.append(Selement[5])

    self.MSE = min(SMSE)
    self.TC = max(STC)
    self.Exact = max(SExact)
    self.Precis = max(SPrecis)
    self.Rec = max(SSens)
    self.F1 = max(SF1)

    if self.fpriority == 'MSE':
      SW = np.where(SMSE == self.MSE) 
    elif self.fpriority == 'TdC':
      SW = np.where(STC == self.TC)
    elif self.fpriority == 'Exactitud':
      SW = np.where(SExact == self.Exact)
    elif self.fpriority == 'Precision':
      SW = np.where(SPrecis == self.Precis)
    elif self.fpriority == 'Recall':
      SW = np.where(SSens == self.Rec)
    elif self.fpriority == 'F1':
      SW = np.where(SF1 == self.F1)
    else:
      SW = np.where(SPrecis == self.Precis)

    SBestPos = SW[0]
    BestK = 3 + 2 * (SBestPos - 1)
    return(BestK)
  
  def get_precision(self):
    return(self.Precis)

  def get_stadistict(self, GSName):
    if GSName == 'MSE':
      GSResult = self.MSE
    elif GSName == 'TdC':
      GSResult = self.TC
    elif GSName == 'Exactitud':
      GSResult = self.Exact
    elif GSName == 'Precision':
      GSResult = self.Precis
    elif GSName == 'Recall':
      GSResult = self.Rec
    elif GSName == 'F1':
      GSResult = self.F1
    else:
      GSResult = self.Precis
    return(GSResult)

Xres, Xval, Yres, Yval = train_test_split(X, Y, test_size = 0.10)

DBsize = len(Xres)
DBp = DBsize / 5
VDB = int(DBp)
CDB = int(DBp * 2)
SDB = int(DBp * 3)
ODB = int(DBp * 4)

Xtest1, Xtrain1, Ytest1, Ytrain1 = train_test_split(Xres, Yres, test_size = 0.80, shuffle = False)

Xtrain2, Ytrain2 = Xres[:VDB].tolist() + Xres[CDB:].tolist(), Yres[:VDB].tolist() + Yres[CDB:].tolist()
Xtrain2, Ytrain2 = np.array(Xtrain2), np.array(Ytrain2)
Xtest2, Ytest2 = Xres[VDB:CDB], Yres[VDB:CDB]

Xtrain3, Ytrain3 = Xres[:CDB].tolist() + Xres[SDB:].tolist(), Yres[:CDB].tolist() + Yres[SDB:].tolist()
Xtrain3, Ytrain3 = np.array(Xtrain3), np.array(Ytrain3)
Xtest3, Ytest3 = Xres[CDB:SDB], Yres[CDB:SDB]

Xtrain4, Ytrain4 = Xres[:SDB].tolist() + Xres[ODB:].tolist(), Yres[:SDB].tolist() + Yres[ODB:].tolist()
Xtrain4, Ytrain4 = np.array(Xtrain4), np.array(Ytrain4)
Xtest4, Ytest4 = Xres[SDB:ODB], Yres[SDB:ODB]

Xtrain5, Xtest5, Ytrain5, Ytest5 = train_test_split(Xres, Yres, test_size = 0.20, shuffle = False)

model1 = KNN()
model1.fit(Xtrain1, Ytrain1, Xtest1, Ytest1, fEpocasP = 5, fpriorityP = 'Precision')
print('')
model2 = KNN()
model2.fit(Xtrain2, Ytrain2, Xtest2, Ytest2, fEpocasP = 5, fpriorityP = 'Precision')
print('')
model3 = KNN()
model3.fit(Xtrain3, Ytrain3, Xtest3, Ytest3, fEpocasP = 5, fpriorityP = 'Precision')
print('')
model4 = KNN()
model4.fit(Xtrain4, Ytrain4, Xtest4, Ytest4, fEpocasP = 5, fpriorityP = 'Precision')
print('')
model5 = KNN()
model5.fit(Xtrain5, Ytrain5, Xtest5, Ytest5, fEpocasP = 5, fpriorityP = 'Precision')

Stadistic1 = model1.get_stadistict('Precision')
Stadistic2 = model2.get_stadistict('Precision')
Stadistic3 = model3.get_stadistict('Precision')
Stadistic4 = model4.get_stadistict('Precision')
Stadistic5 = model5.get_stadistict('Precision')

StadisticFin = (Stadistic1 + Stadistic2 + Stadistic3 + Stadistic4 + Stadistic5) / 5

print('Precisión 1 =', str(Stadistic1) + '.')
print('Precisión 2 =', str(Stadistic2) + '.')
print('Precisión 3 =', str(Stadistic3) + '.')
print('Precisión 4 =', str(Stadistic4) + '.')
print('Precisión 5 =', str(Stadistic5) + '.')
print('Precisión final =', str(StadisticFin) + '.')

YvalPred = []
for element in Xval:
  YvalPred.append(model1.PrednewPoint(element))

from tabulate import tabulate

ModelStatistics = Estadisticas(YvalPred, Yval)

STable = [ ['MSE', str(ModelStatistics[0])],
     ['Tasa de clasificación', str(ModelStatistics[1])],
     ['Exactitud', str(ModelStatistics[2])],
     ['Precisión', str(ModelStatistics[3])],
     ['Recall', str(ModelStatistics[4])],
     ['F1', str(ModelStatistics[5])] ]

print(tabulate(STable, headers = ['Métrica', 'Valor'], tablefmt="presto"))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(Yval, YvalPred, labels = np.unique(Yval))
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = np.unique(Yval))
disp.plot()
plt.show()

