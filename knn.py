import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report,accuracy_score,recall_score
from sklearn.metrics import accuracy_score, f1_score

veriSeti = pd.read_csv("dataR2.csv")
#ÖNİŞLEME
veriSeti=veriSeti.rename(columns={"Classification":"Karar"})
veriSeti["Karar"].value_counts()
veriSeti["Karar"]=np.where(veriSeti["Karar"]==1,"Sağlıklı","Kanser")
veriSeti["Karar"].value_counts()

veriSeti.Karar = veriSeti.Karar.astype("category")
pd.set_option("display.max_columns",20)
veriSeti.describe(include="all")
veriSeti.dtypes
print(veriSeti.isnull().sum())
my_cors = pd.DataFrame(np.corrcoef(veriSeti.iloc[:,0:9],rowvar=False).round(2),columns=veriSeti.columns[0:9])
my_cors.index=veriSeti.columns[0:9]
sns.heatmap(my_cors,annot=True,square=True,cmap=sns.color_palette("flare",as_cmap=True))  #hedef nitelik hariç korelasyon haritası

x_train,x_test,y_train,y_test = train_test_split(veriSeti.iloc[:,0:9],veriSeti.iloc[:,9],test_size=0.3,random_state=1) # x tahmini,y hedef nitelik
scaler = MinMaxScaler()
x_train_n = pd.DataFrame(scaler.fit_transform(x_train),columns=x_train.columns) #eğitim verisetindeki parametreleri(min max oluştur)
x_test_n = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)
x_train_n.describe()

#MODELLEME
knn_modeli = KNeighborsClassifier(n_neighbors=5,metric="euclidean") # komşu sayısı 5,öklit fonk
knn_modeli.fit(x_train_n,y_train)
#PERFORMANS DEĞERLENDİRME
y_tahmin = knn_modeli.predict(x_test_n)
print("k-NN Modeli Tahminleri: ",y_tahmin[0:5])
print("Gerçek Değerler: ",np.array(y_test[0:5]))
my_cm = confusion_matrix(y_test, y_tahmin,labels=["Sağlıklı","Kanser"])
my_cm
my_cm_p = ConfusionMatrixDisplay(my_cm,display_labels=["Sağlıklı","Kanser"])
my_cm_p.plot()

tn,fp,fn,tp = my_cm.ravel() #doğru pozitif,yanlış pozitif ayarlama
print("True Negatives: ",tn)
print("False Negative: ",fn)
print("True Positives: ",tp)
print("False Positives: ",fp)

dogruluk1 = (tp+tn)/(tp+tn+fn+fp)

rapor = classification_report(y_test, y_tahmin, labels=["Sağlıklı","Kanser"])
print(rapor)

#K Katsayıları Performans Grafiği
dogruluk = []
fOlcusu = []
k = range(2,21) #2 -20 arası tüm sayıları k katsayısı için dene grafik yap
for i in k:
    knn_modeli = KNeighborsClassifier(n_neighbors=i,metric="euclidean")
    knn_modeli.fit(x_train_n, y_train)
    y_tahmin= knn_modeli.predict(x_test_n)
    dgrlk = accuracy_score(y_test, y_tahmin).round(4)
    
    fOlc = f1_score(y_test, y_tahmin, average="binary", pos_label="Kanser").round(4)
    dogruluk.append(dgrlk)
    fOlcusu.append(fOlc)
    
    plt.plot(k,dogruluk,'bx-')
    plt.xticks(k)
    plt.title("K-NN Model Performansı")
    plt.xlabel("k Komşu Sayısı")
    plt.ylabel("Doğruluk")
    plt.show()
    
