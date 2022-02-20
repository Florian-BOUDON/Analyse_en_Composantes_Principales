# importation des librairies nécessaires 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn import decomposition
from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Création d'un DF qui donne tous les DF, liste et np.array et leurs significations
id =["coord","eigval","ratio_eigval","df_bs","df_ctr","cos2 ", "df_di", "df_corr",
     "df_cos2var", "df_ctrvar" ,"df_barycentre"]
meta_df = pd.DataFrame({'Commentaires': 
                        ['Coordonnées des individus sur les axes factorielles',
                         'Valeur propre = Variance composante principale',
                         'Somme des variances expliqués = Somme des valeurs propres',
                         "Méthode des batons",
                         "Contribution des individus aux axes factorielles",
                         "Qualité de la représentation des individus dans le plan factoriel",
                         "Contribution des individus dans l'inertie totale",
                         "DF des corrélations entre les variables et les composantes  ",
                         "Qualité de la représentation des variables",
                         "Contribution de la variable  à la composante",
                         "Barycentre de chaque groupes"]}, index=id )

# Le DF doit avoir un index    
# On nomme le DF X  
X = pd.read_csv("data_acp_cours_complet.csv",sep=";", index_col=0)
n=X.shape[0]
p=X.shape[1]

# instanciation & données centrées reduites
sc=StandardScaler()
Z=sc.fit_transform(X)
print(Z)

# Crétion de l'ACP
acp=PCA(svd_solver='full')
coord = acp.fit_transform(Z)

# Variance expliqué & Valeurs propres
eigval = (n-1)/n*acp.explained_variance_

# proportion de variance expliquée
ratio_eigval = acp.explained_variance_ratio_

# screen plot
plt.plot(np.arange(1,p+1),eigval)
plt.title("Screen plot")
plt.ylabel("Eigen values")
plt.xlabel("Factor number")
plt.show()

# Cumul de variance expliquée
plt.plot(np.arange(1,1+p), np.cumsum(acp.explained_variance_ratio_))
plt.title("Explained variance vs. # of factors")
plt.ylabel("Cumsum explained variance ratio")
plt.xlabel('Factor number')
plt.show()

# seuils pour test des bâtons brisés
bs = 1/np.arange(p,0,-1)
bs = np.cumsum(bs)
bs = bs[::-1]
df_bs = pd.DataFrame({'Val.Propre':eigval, 'Seuils':bs})

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# Espace des Individus

# Positionnement des individus dans le premier plan
fig,axes = plt.subplots(figsize=(10,10))
axes.set_xlim(-6,6)
axes.set_ylim(-6,6)
# Placement des étiquettes et des observations
for i in range(n):
    plt.annotate(X.index[i],(coord[i,0],coord[i,1]))    
# ajouter des axes
plt.plot([-6,6],[0,0],color='silver',linestyle='--',linewidth=1)
plt.plot([0,0],[-6,6],color='silver', linestyle='--', linewidth=1)
# affichage
plt.show()
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# Contribution des individus dans l'inertie totale
di=np.sum(Z**2,axis=1)
df_di = (pd.DataFrame({'d_i':di},index=X.index))

# qualité de la représentation des individus  COS2
cos2=coord**2
for j in range(p):
    cos2[:,j]=cos2[:,j]/di   
cos2 = pd.DataFrame({'cos2_1':cos2[:,0],'cos2_2':cos2[:,1]},index=X.index)

# Contribution aux axes
ctr=coord**2
for j in range(p):
    ctr[:,j]=ctr[:,j]/(n*eigval[j])
df_ctr = pd.DataFrame({'CTR_1':ctr[:,0],'CTR_2':ctr[:,1]},index = X.index)


#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# Espace des Variables

# racine carré des valeurs propres
sqrt_eigval = np.sqrt(eigval)
# corrélation des variables avec les axes
corvar = np.zeros((p,p ))
for k in range(p):
    corvar[: , k] = acp.components_[k, :]*sqrt_eigval[k]
# DF des corrélations entre les variables et les composantes 
df_corr = pd.DataFrame({'COR_1':corvar[:,0],'COR_2':corvar[:,1]},index = X.columns)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# cercle des correlations
fig,axes=plt.subplots(figsize=(8,8))
axes.set_xlim(-1,1)
axes.set_ylim(-1,1)
# Affichage des étiquettes (noms des variables)
for j in range(p):
    plt.annotate(X.columns[j],(corvar[j,0],corvar[j,1]))
    plt.plot([0,corvar[j,0]],[0,corvar[j,1]],color='blue',alpha=0.2)  
# ajouter les axes
plt.plot([-1,1],[0,0],color='silver',linestyle='--',linewidth=1)
plt.plot([0,0],[-1,1],color='silver',linestyle='--',linewidth=1)
# Ajouter un cercle
cercle = plt.Circle((0,0),1,color='blue',fill=False)
axes.add_artist(cercle)
# affichage
plt.show()
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# cosinus carré des variables: corrélation entre la variable et l'axe
cos2var=corvar**2
df_cos2var = pd.DataFrame({'COS2_1':cos2var[:,0],'COS2_2':cos2var[:,1]},
                          index = X.columns )

# contributions des variables aux 2 premiers axes factorielles
ctrvar = cos2var
for k in range(p):
    ctrvar[:,k] = ctrvar[:,k]/eigval[k]
# on n'affiche que les 2 premiers axes
df_ctrvar = pd.DataFrame({'CTR_1':ctrvar[:,0],'CTR_2':ctrvar[:,1]},index =X.columns )
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# Le DF contenant les individus supplémentaires, soit les lignes,
# doivent conteneir exactement le bon nombre de variables
# Ce DF est appelé ind_sup

ind_Sup

# centrage et reduction avec les paramètres des individus actifs
Zind_Sup=sc.transform(ind_Sup)
print(Zind_Sup)

# projection dans l'espace factoriel
coordSup=acp.transform(Zind_Sup)

# positionnement des individus supplémentaires dans le premeier plan
fig,axes=plt.subplots(figsize=(10,10))
axes.set_xlim(-6,6)
axes.set_ylim(-6,6)

# étiquette des points actifs
for i in range(n):
    plt.annotate(X.index[i],(coord[i,0],coord[i,1]))

# etiquette des points supplémmentaires (illustratif) en bleu
for i in range(coordSup.shape[0]):
    plt.annotate(ind_Sup.index[i],(coordSup[i,0],coordSup[i,1]),color='blue')
    
# ajouter des axes
plt.plot([-6,6],[0,0],color='silver',linestyle='--',linewidth=1)
plt.plot([0,0],[-6,6],color='silver',linestyle='--',linewidth=1)

# affichage
plt.show()
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# Le DF contenant les variables supplémentaires, soit les colonnes,
# doit conteneir exactement le bon nombre de individus
# Ce DF est appelé Var_Sup

# on reccupère les variables quantitatives dans un format np
vsQuanti= Var_Sup.iloc[:,:2].values
# Corrélation avec les axes fatoriels
corSupp = np.zeros((vsQuanti.shape[1],p))
for k in range(p):
    for j in range(vsQuanti.shape[1]):
        corSupp[j,k]=np.corrcoef(vsQuanti[:,j],coord[:,k])[0,1]
df_corSupp      

# cercle des corrélations avec var supp
fig,axes = plt.subplots(figsize=(8,8))
axes.set_xlim(-1,1)
axes.set_ylim(-1,1)
# variables actives
for j in range (p):
    plt.annotate(X.columns[j],(corvar[j,0],corvar[j,1]))
    plt.plot([0,corvar[j,0]],[0,corvar[j,1]],color='blue',alpha=0.2)
    # variables illustratives
for j in range(vsQuanti.shape[1]):
    plt.annotate(Var_Sup.columns[j],(corSupp[j,0],corSupp[j,1]),color='g')
    plt.plot([0,corSupp[j,0]],[0,corSupp[j,1]],color='g',alpha=0.2)
# ajouter les axes
plt.plot([-1,1],[0,0],color='silver',linestyle='--',linewidth=1)
plt.plot([0,0],[-1,1],color='silver',linestyle='--',linewidth=1)
# ajouter un cercle
cercle = plt.Circle((0,0),1,color='b',fill=False)
axes.add_artist(cercle)
# affichage
plt.show()

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# Variables Qualitatives
# Il s'agit de metttre en évidence l'effet de groupe 
# La vriable quali doit être traité à part du DF de base

# on reccupère les variables sup
Var_Sup=pd.read_csv("variables_sup_voiture_acp.csv",sep=";",index_col=0)
# traitement de variable qualitative supplémentaire
vsQuali=Var_Sup.iloc[:,2]

#liste des couleurs
couleurs = ['r','g','b']

#faire un graphique en coloriant les points
fig, axes = plt.subplots(figsize=(10,10))
axes.set_xlim(-4,6)
axes.set_ylim(-4,4)

#pour chaque modalité de la var. illustrative
for c in range(len(modalites)):
#numéro des individus concernés
    numero = np.where(vsQuali == modalites[c])
#les passer en revue pour affichage
    for i in numero[0]:
        plt.annotate(X.index[i],(coord[i,0],coord[i,1]),color=couleurs[c])
        
#ajouter les axes
plt.plot([-6,6],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-6,6],color='silver',linestyle='-',linewidth=1)

#affichage
plt.show()

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# structure intermédiaire
df_barycentre = pd.DataFrame({'Finition':vsQuali,'F1':coord[:,0],'F2':coord[:,1] })

# on calcul les moyennes conditionnelles
df_barycentre = df_barycentre.pivot_table(index='Finition',values=['F1','F2'],aggfunc=pd.Series.mean)






