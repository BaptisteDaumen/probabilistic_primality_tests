import random as rd
import numpy as np
import math as m
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#%%
# Génération aléatoire d'un nombre de taille n
def genere_nombre_impair(n):
    a = rd.randint(10**(n-1),10**n-1)
    if n == 1:
        a = rd.randint(2,7)
        if a % 2 == 0:
            return(a+1)
        else:
            return(a)
    elif a % 2 == 0:
        return(a+1)
    else:
        return(a)

# Exponentiation rapide
    
def exponentiation(n,i):
    if i == 0:
        return(1)
    elif i%2 == 0:
        return(exponentiation(n*n,i//2))
    else:
        return(exponentiation(n*n,i//2)*n)

# Nombre de Mersenne
        
def mersenne(n):
    return(exponentiation(2,n)-1)
        
# Test déterministe de primalité : complexité en O(sqrt(n))
    

    
def est_premier_1(n):
    if n == 1 or n == 0:
        return False
    if n == 2:
        return True
    k = 2
    while k**2 <= n:
        if n%k == 0:
            return False
        k = k+1
    return True
        
#%%
def test_fermat(n,a):
    assert a < n
    if pow(a,n-1,n)==1:
        return(True) # probablement premier
    else:
        return(False) # sûrement composé
        
def test_fermat_precision(n,pr):
    if pr > n-1:
        for a in range(1,n):
            if not test_fermat(n,a):
                return(False)
        return(True) 
    else:
        while pr > 0 :
            a = rd.randint(1,n-1)
            if not test_fermat(n,a):
                return(False)
            pr = pr - 1
        return(True)
        
def pgcd(a,b):
    if b==0:
        return a
    else:
        r=a%b
        return pgcd(b,r)
        
def nombre_carmichael(n):
    for a in range(1,n):
        if pgcd(a,n) == 1:
            if test_fermat(n,a) is False :
                print(a)
                return (False)
    return(True)


def genere_premier_taille_verif_fermat(nb,n,pr):
    res = []
    nb_tot = nb
    vrai = 0
    while nb > 0:
        a = genere_nombre_impair(n)
        if test_fermat_precision(a,pr):
            if est_premier(a):
                res.append((a,True))
                vrai = vrai + 1
            else:
                res.append((a,False))
            nb = nb - 1
    return(res,vrai,vrai/nb_tot)
    
# 2D

def graphique_fermat_2D(nbr_genere,taille,pr_max):
    x,y = [],[]
    for i in range(1,pr_max+1):
        x.append(i)
        y.append(genere_premier_taille_verif_fermat(nbr_genere,taille,i)[2])
    plt.plot(x,y)
    plt.title("Ordre de grandeur 10⁹ : 10000 essais")
    plt.xlabel("Itérations")
    plt.ylabel("Efficacité")
    plt.show()



"""graphique_fermat_2D(10000,9,20)"""

#%%
# decomposition n = 2^s*d+1
def decomposition(n):
    s = 0 
    d = n-1
    while d%2 == 0:
        d = d//2
        s = s + 1
    return(s,d)
    
# second test dans Miller-Rabin
    
def second_test(n,s,d,b):
    for i in range(s):
        if b == n-1:
            return(True)
        b = pow(b,2,n)
    return(False)
            
def miller_rabin(n,a,s,d): # n doit etre impair
    assert n%2 == 1
    #(s,d) = decomposition(n)
    b = pow(a,d,n)
    if b == 1 or second_test(n,s,d,b):
        return(True) # n passe le test
    else:
        return(False) # n compose

def miller_rabin_precision(n,pr): # precision < n (montrer qu'on peut se ramener à [|1,n-1|])
    (s,d) = decomposition(n)
    while pr > 0 :
        a = rd.randint(1,n-1)
        if not miller_rabin(n,a,s,d):
            return(False)
        pr = pr - 1
    return (True)
    

# genere un nombre "presque" premier avec la probabilite ??? qu'il soit premier

def genere_premier_taille_n(nb,n,pr):
    res = []
    while nb > 0:
        a = genere_nombre_impair(n)
        if miller_rabin_precision(a,pr):
            res.append(a)
            nb = nb - 1
    return(res)

        
def est_premier(n):
    for a in range(1,int(2*m.log(n)**2)+1):
        s, d = decomposition(n)
        if not miller_rabin(n,a,s,d):
            return(False)
    return(True)

def genere_premier_taille_verif_miller_rabin(nb,n,pr):
    res = []
    nb_tot = nb
    vrai = 0
    while nb > 0:
        a = genere_nombre_impair(n)
        if miller_rabin_precision(a,pr):
            if est_premier(a):
                res.append((a,True))
                vrai = vrai + 1
            else:
                res.append((a,False))
            nb = nb - 1
    return(res,vrai,vrai/nb_tot)

# 3D

def z(v_x,v_y,taille,nbr_gen):
    (n,p) = np.shape(v_x)
    z = np.zeros((n,p))
    for i in range(n):
        for j in range(p):
            z[i,j] = genere_premier_taille_verif_miller_rabin(nbr_gen,taille,v_y[i,j])[2]
    return(z)
    
def graphique_miller_rabin_3D(nbr_essai_par_pr,taille,pr,nbr_gen):
    x1 = np.linspace(1, nbr_essai_par_pr, nbr_essai_par_pr)
    y1 = np.linspace(1, pr, pr)

    v_x, v_y = np.meshgrid(x1, y1)

    v_z = z(v_x,v_y,taille,nbr_gen)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(v_x, v_y, v_z, cmap='Blues');
    

# 2D

def graphique_miller_rabin_2D(nbr_genere,taille,pr_max):
    x,y = [],[]
    for i in range(1,pr_max+1):
        x.append(i)
        y.append(genere_premier_taille_verif_miller_rabin(nbr_genere,taille,i)[2])
    plt.show()
    plt.plot(x,y)
    plt.title("Ordre de grandeur 10⁹: 100000 essais")
    plt.xlabel("Itérations")
    plt.ylabel("Efficacité")
    
"""print(graphique_miller_rabin_2D(10000,9,20))"""

#%%
def symbole_legendre(n,a):
    if n==1 or a==1 or n==0 or a==0:
        return(1)
    elif a == 2:
        if ((n*n-1)//8)%2==0:
            return(1)
        else:
            return(-1)
    elif a%n == 0:
        return(0)
    elif a%2 == 0:
        a = a//2
        if ((n*n-1)//8)%2==0:
            return(symbole_legendre(n,a))
        else:
            return((-1)*symbole_legendre(n,a))
    else:
        b = ((n-1)*(a-1)//4)
        n = n%a
        if b%2 == 0 :
            return(symbole_legendre(a,n))
        else:
            return((-1)*symbole_legendre(a,n))
        
def test_solovay_strassen(n,a):
    s = symbole_legendre(n,a)
    if s == -1:
        s = n-1
    c = pow(a,(n-1)//2,n)
    if c == s and s != 0:
        return (True) # passe le test
    else:
        return (False) # n est compose
    
def test_solovay_strassen_precision(n,pr):
    while pr > 0:
        a = rd.randint(1,n-1)
        if not test_solovay_strassen(n,a):
            return (False)
        pr = pr - 1
    return(True)
    
def genere_premier_taille_verif_solovay_strassen(nb,n,pr):
    res = []
    nb_tot = nb
    vrai = 0
    while nb > 0:
        a = genere_nombre_impair(n)
        if test_solovay_strassen_precision(a,pr):
            if est_premier(a):
                res.append((a,True))
                vrai = vrai + 1
            else:
                res.append((a,False))
            nb = nb - 1
    return(res,vrai,vrai/nb_tot)
    
# 2D

def graphique_solovay_strassen_2D(nbr_genere,taille,pr_max):
    x,y = [],[]
    for i in range(1,pr_max+1):
        x.append(i)
        y.append(genere_premier_taille_verif_solovay_strassen(nbr_genere,taille,i)[2])
    plt.plot(x,y)
    plt.title("Ordre de grandeur 10⁹: 100000 essais")
    plt.xlabel("Itérations")
    plt.ylabel("Efficacité")
    plt.show()
    

"""print(graphique_solovay_strassen_2D(10000,9,20))"""
#%%
# Resultat des tests
    
def graphique_comparaison_general(nbr_genere,taille,pr_max):
    plt.show()
    x, y_m, y_s, y_f = [], [], [], []
    for i in range(1,pr_max+1):
        x.append(i)
        y_m.append(genere_premier_taille_verif_solovay_strassen(nbr_genere,taille,i)[2])
        y_s.append(genere_premier_taille_verif_miller_rabin(nbr_genere,taille,i)[2])
        y_f.append(genere_premier_taille_verif_fermat(nbr_genere,taille,i)[2])
    plt.plot(x, y_m, color = "green")
    plt.plot(x,y_s, color = "red")
    plt.plot(x,y_f, color = "blue")
    plt.legend(["Miller-Rabin", "Solovay-Strassen", "Fermat"])
    plt.title(" Ordre de grandeur 10⁹: 10000 essais")
    plt.xlabel("Précision")
    plt.ylabel("Efficacité")
    plt.show()
    
#%%

def temoin_miller_rabin(n):
    res = 0
    s,d = decomposition(n)
    for a in range(1,n):
        if miller_rabin(n,a,s,d):
            res = res + 1
    return(res/(n-1))
    
def liste_nombre_premier(n):
    res = []
    for i in range(3,n):
        if est_premier_1(i):
            res.append(i)
    return res
    

def resultat_miller_rabin(n):
    res = []
    liste = liste_nombre_premier(n)
    indice = []
    M = 0
    for i in liste:
        for j in liste:
            m = temoin_miller_rabin(i*j)
            res.append(m)
            if m > M:
                indice.append(i*j)
                M = m
    return(res, max(res), indice)
    


#print(temoin_miller_rabin(146611))


# resultat miller_rabin : [87, 145, 217, 341, 703, 1891, 12403, 38503, 79003, 88831]

# resultat solovay_strassen : [21, 25, 91, 703, 1729]

def decomposition_premier(n):
    res = []
    for i in range(2,n):
        if n%i == 0:
            res.append(i)
    return(res)
"""
print(temoin_miller_rabin(88831))
print(decomposition_premier(88831))
print(liste_nombre_premier(50000))
"""
def decomposition_analyse(l):
    res = []
    s_res = []
    for e in l:
        l_d = decomposition_premier(e)
        for i in l_d:
            s_res.append(decomposition(i))
        s_res.append(decomposition(e))
        res.append(s_res)
        s_res = []
    return res

l_mr = [703, 1891, 12403, 38503, 79003, 88831]
"""
print(decomposition_analyse(l_mr))

print(temoin_miller_rabin(88831))
"""
#%%
def temoin_solovay_strassen(n):
    res = 0
    for a in range(1,n):
        if test_solovay_strassen(n,a) :
            res = res + 1
    return (res/(n-1))

def resultat_solovay_strassen(d,f):
    res = []
    indice = []
    M = 0
    c = 0
    for i in range(d,f,2):
        if c == 100:
            c = 0
            print(i)
        if not est_premier(i):
            m = temoin_solovay_strassen(i)
            res.append(m)
            if m >= M:
                indice.append(i)
                M = m
                print(M)
        c = c + 2
        
    return(res, max(res), indice)
    

l_ss = [703, 1729, 15841, 46657]

"""print(decomposition_analyse(l_mr))"""

def temoin_liste_mr(l):
    res = []
    for e in l:
        res.append(temoin_miller_rabin(e))
    return(res)
    
def temoin_liste_ss(l):
    res = []
    for e in l:
        res.append(temoin_solovay_strassen(e))
    return(res)

"""print(temoin_liste_mr(l_mr))"""
"""print(temoin_liste_ss(l_ss))"""

#algorithme exponentiation modulaire
    
def expmodrap(a, e, n):    
    p=1    
    while e>0:        
        if e % 2 == 1:            
            p = (p*a)%n       
        a=(a*a)%n       
        e=e//2    
    return p
