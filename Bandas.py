import numpy as np
from matplotlib import pyplot as plt
from scipy.special import jn, yv
from scipy.special import hankel1 as hn
from numpy.linalg import inv, pinv, norm, det
from math import factorial as sfactorial
from scipy.optimize import newton, fsolve, root, show_options
import os

###########################################################################
##################### Suma de red #########################################
###########################################################################

def kron(i, j):
    """kron(i,j) entrega el delta de kronecker i,j, 1 si i==j, 0 si i!=j."""
    if i == j:
        return 1
    else:
        return 0

def K(a, k, lattice = 'sq'):
    '''
    K(a, k, lattice) = [float, float] entrega el vector de Bloch que recorre
    la red recíproca de una red del tipo 'lattice', con constante de red 'a',
    pasando por los puntos de alta simetría correspondientes a dicha red, los
    parámetros que recibe:

    a: 'float' constante de red
    k: 'float' magnitud del vector de onda
    lattice: 'str' tipo de red que admite el sistema, puede ser:
        'sq' = red cuadrada
        'hx' = red hexagonal
    '''

    #aca comprobamos que el vector de onda este dentro del rango permitido

    if k > 3*np.pi/a:
        raise ValueError('k outside of the Brillouin zone')

    #Aca defino las condiciones y los intervalos de mi funcion por partes y
    #la forma del vector en los int.

    if lattice == 'hx':
        cond = [k <= (np.pi / a), (k > (np.pi / a) and k <= (2 * np.pi / a)), k > (2 * np.pi /a)];
        int1 = [-k/2 + np.pi/(2*a), 0]
        int2 = [3*k/4 - 3*np.pi/(4*a), np.sqrt(3)*k/4 - np.pi*np.sqrt(3)/(4*a)]
        int3 = [k/4 + np.pi/(4*a), -np.sqrt(3)*k/4 + 3*np.pi*np.sqrt(3)/(4*a)]

    elif lattice == 'sq':
        cond = [k <= (np.pi / a), (k > (np.pi / a) and k <= (2 * np.pi / a)), k > (2 * np.pi /a)];
        int1 = [np.pi / a - k, np.pi / a - k];
        int2 = [k - np.pi / a, 0];
        int3 = [np.pi / a, k - 2 * np.pi / a];

    bloch = [np.array(int1), np.array(int2), np.array(int3)]

    for i in range(len(cond)):
        if cond[i]:
            return bloch[i]

def Kh(a, k1, k2, lattice='sq'):
    """
    Kh(a, k1, k2) = [float, float] entrega el vector de red reciproca que
    lleva desde la celda central a una celda que se encuentra k1 veces en la
    direccion del primer vector primitivo de red reciproca y k2 veces en la
    direccion del segundo vector primitivo de red reciproca de
    una red del tipo 'lattice' de constante a.

    a: 'float' constante de red
    k1: 'integer' indice de movimiento en la direccion del primer vector unita
        -rio de la red reciproca
    k2: 'integer' indice de movimiento en la direccion del segundo vector unita
        -rio de la red reciproca
    lattice: 'str' tipo de red que admite el sistema, puede ser:
        'sq' = red cuadrada
        'hx' = red hexagonal
    """

    if lattice == 'sq':

        vec_1 = [2 * np.pi / a, 0];
        vec_2 = [0, 2 * np.pi / a];


    elif lattice == 'hx':

        vec_1 = [0, np.pi*np.sqrt(3)/a]
        vec_2 = [3*np.pi/(2*a), np.pi*np.sqrt(3)/(2*a)]

    vec = k1*np.array(vec_1) + k2*np.array(vec_2);

    return np.array(vec)

def S1(M, m, k, n, a, k0_, lattice='sq'):
    """
     S1 entrega parte de la suma sobre los vectores de red reciproca
     necesario para el calculo de la suma de red, todos los inputs son
     escalares, M y m son modos, f es la frecuencia, pol es la polarizacion
     del modo, y n la cantidad de veces que queremos iterar sobre la suma

     M: 'integer' modo de entrada
     m: 'integer' modo de salida
     k: 'float' modulo del vector de onda de bloch, acotado en el intervalo
        [0, 3pi]
     n: 'integer' cantidad de iteraciones q hace la suma.
     a: 'float' constante de red
     k0_: 'complex' modulo vector de onda
     lattice: 'str' tipo de red que admite el sistema, puede ser:
        'sq' = red cuadrada
        'hx' = red hexagonal
     """

    N0 = M - m;
    N = abs(N0);
    #Inicializacion de la suma
    S = 0;


    for i in range(-n, n + 1):
        for l in range(-n, n + 1):

            Qh = Kh(a, i, l, lattice) + K(a, k, lattice)
            Qh_ = norm(Qh);
            x_pos = Qh[0];
            y_pos = Qh[1];
            comp = x_pos + 1j * y_pos;
            ang = np.angle(comp);

            if Qh_ == 0:
                pass;
            else:

                num = jn(N+1, Qh_)*np.exp(1j*N*ang)
                den = Qh_*(Qh_**2 - k0_**2)

                S = S + num/den

    if lattice == 'sq':
        area = 1
    elif lattice == 'hx':
        area = 3*np.sqrt(3)/2

    val = S*k0_*4*(1j)**(N+1)/area

    return val

def S(M, m, k, n, a, k0_, lattice='sq'):
    """
    S entrega la suma de red del sistema descrito para
    la diferencia de modos M-m, una polarizacion pol, en el vector de bloch
    k y una frecuencia f, sumando n veces.

    M: 'integer' modo de entrada
    m: 'integer' modo de salida
    k: 'float' modulo del vector de onda de bloch, acotado en el intervalo
       [0, 3pi]
    n: 'integer' cantidad de iteraciones q hace la suma.
    a: 'float' constante de red
    k0_: 'complex' modulo vector de onda
    lattice: 'str' tipo de red que admite el sistema, puede ser:
       'sq' = red cuadrada
       'hx' = red hexagonal
    """

    N0 = M - m;
    K_ = norm(K(a, k));

    # Primero calculemos el valor con el valor absoluto de N, luego compruebo si N0 es positivo (mantiene la forma)
    # Si N0 negativo, calculamos -S*

    N = abs(N0);

    v1 = ((2j + k0_*np.pi*hn(1, k0_))/(k0_*np.pi))*kron(0,N)
    v2 = S1(M, m, k, n, a, k0_,lattice)

    S = (-1/jn(N+1, k0_))*(v1+v2)

    if N0 >= 0:
        pass;
    if N0 < 0:
        S = ((-1))*np.conj(S);

    return S

###########################################################################
#################### Utilidades ###########################################
###########################################################################
 

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def savefrec(k2, frec, path):
    count = 0;
    for i in k2:
        name = path + '/frecuencias' + str(np.around(i,2)) + '.txt'
        np.savetxt(name, frec[count]);
        count = count + 1;

def path(lattice):
    """crea el path para crear la carpeta en la que se guardaran los datos
    con el nombre x=filling, N= numero de divisiones en el camino de k, tol=
    tolerancia con la que se calculan las autofrecuencias"""

    path = os.path.join(os.path.expanduser('~'), 'Documents', 'Metamateriales', 'python','lattice ='+ str(lattice))

    return path

def readfrec(k, path0, tol, N):
    '''Lee los archivos guardados de frecuencias para kada elemento de k, path es un string que contiene el path
    donde estan guardados los archivos, N es la cantidad de bandas que queremos entregar'''

    omega = np.zeros((len(k), N, 2))
    count1 = 0;
    for i in k:
        name = path0 + '/frecuencias' + str(np.around(i,2)) + '.txt'
        frec = np.loadtxt(name)

        #Aquí ordena las frecuencias

        omega[count1,:,:] = frec
        count1 = count1 + 1

    return omega

def repetido(valor, lista):
    for l in range(len(lista)):
        if np.isclose(valor, lista[l]):
            return  True
    return False

def n_sol(list):
    n = 0;
    for i in range(len(list)):
        if list[i] != 0:
            n = n + 1
    return n

def ordFrec(f,k,N):
    omega = np.zeros((len(k), N, 2));
    for i in range(len(k)):
        omega[i,:,0] = np.sort(f[i,:,0])
    return omega

def convParam(young, poisson):
    '''convParam(young, poisson) convierte los parametros elasticos desde el
    modulo de young y el coeficiente de poisson, a los modulos de lame y shear
    entregando un vector [lame, shear]'''

    mu = young/(2*(1+poisson))
    lame = young/((1+poisson)*(1-2*poisson))
    elastic_param = [lame, mu]

    return elastic_param


###########################################################################
####################### Elastic Systems ###################################
###########################################################################

class Red:
    """En la clase Red, entregamos todos los parametros que puede tener nuestro
    sistema, como el nombre de los compuestos, la forma de la red (cuadrada,
    hexagonal, etc.), las velocidades de las ondas en los compuestos, etc.
    La forma para entregas los parametros es la siguiente:
    - self.comp: string
        Este es un atributo para identificar los materiales que componen el
        sistema, la matriz y los scatterers cilindricos
    - self.vel0 = [vel0 long, vel0 transv]
        Este atributo es una lista con las velociades longitudinales y
        transversales en la matriz, entregadas en ese orden.
    - self.vels = [vel0 long, vel0 transv]
        Este atributo es una lista con las velociades longitudinales y
        transversales en el cilindro, entregadas en ese orden.
    - self.dens = [rho0, rhos]
        Este atributo es una lista con las densidades del material de la matriz
        y los cilindros, entregadas en ese orden.
    - self.filling = real
        Radio de llenado, es la relacion entre el area ocupada por los cilindros
        y la matriz
                    x = A_{cilindro transversal} /A_{celda unidad}
    """
    a = 1

    def __init__(self, comp):
        self.comp = comp
        self.vel0 = None
        self.vels = None
        self.dens = None
        self.filling = 0
        self.cut = 2
        self.nbands = 0
        self.nk = 0
        self.pace = 0.5
        self.k_init = 0
        self.k_end = 3*np.pi
        self.shear = None
        self.lame = None
        self.r = None
        self.k = None
        self.omega = None
        self.foldername = None
        self.lattice = 'sq'
        self.frecfolder = None

    def asign_param(self):
        """Modifica los atributos que dependen de otros parametros, como los
        coeficientes de lame y el radio"""

        if self.dens==None:

            raise ValueError('No density inputs')

        elif (self.vel0 == None and self.vels == None and self.shear == None and self.lame == None):

            raise ValueError('No material elasticity parameters given')

        elif not(self.vel0 == None and self.vels == None):

            mu0 = self.dens[0]*(self.vel0[1]**2)
            mus = self.dens[1]*(self.vels[1]**2)
            lamb0 = self.dens[0]*(self.vel0[0]**2)-2*self.shear[0]
            lambs = self.dens[1]*(self.vels[0]**2)-2*self.shear[1]

            self.shear = [mu0, mus]
            self.lame = [lamb0, lambs]

        elif (self.vel0 == None and self.vels == None):
            vl0 = np.sqrt((2*self.shear[0]+self.lame[0])/self.dens[0])
            vt0 = np.sqrt(self.shear[0]/self.dens[0])
            vls = np.sqrt((2*self.shear[1]+self.lame[1])/self.dens[1])
            vts = np.sqrt(self.shear[1]/self.dens[1])

            self.vel0 = [vl0, vt0]
            self.vels = [vls, vts]

        self.r = np.sqrt(self.filling/np.pi)
        self.k = np.linspace(self.k_init, self.k_end, self.nk)
        self.foldername = path(self.lattice)
        self.frecfolder = os.path.join(self.foldername, "filling = " + str(self.filling) + ' N='+ str(self.nbands))

    def k0(self, f, p):
        """f un arreglo de 0x2 donde f[0] es la parte real de la frecuencia y
        f[1] la parte imaginaria, y p el modo de polarizacion, 0 si es
        longitudinal, 1 si es transversal"""

        Cl0, Ct0 = self.vel0

        if p > 1 or p < 0:
            print('Error, polarizacion incorrecta')

        if p == 0:
            v1 = 1/Cl0;
        elif p == 1:
            v1 = 1/Ct0;

        re_f, im_f = f
        w = (re_f + 1j*im_f)

        val = w*v1

        return val

    def ks(self, f, p):
        """f un arreglo de 0x2 donde f[0] es la parte real de la frecuencia y
        f[1] la parte imaginaria, y p el modo de polarizacion, 0 si es
        longitudinal, 1 si es transversal"""

        Cls, Cts = self.vels

        if p > 1 or p < 0:
            print('Error, polarizacion incorrecta')

        if p == 0:
            v1 = 1/Cls;
        elif p == 1:
            v1 = 1/Cts;

        re_f, im_f = f
        w = (re_f + 1j*im_f)

        val = w*v1

        return val

    def D(self, m, f):
        """D(m, f) entrega la matriz de transmision dentro de la celda unidad
        para un modo y frecuencia especificado, donde m es el modo y f la
        frecuencia. Esto solo es valido cuando los scatterers son cilindricos"""

        mu0, mus = self.shear
        lamb0, lambs = self.lame

        def A1l(m, f):

            k = self.k0(f, 0)
            r = self.r

            v1 = -m * jn(m, r*k)
            v2 = k*r*jn(m-1, r*k)

            val = (v1+v2) / r

            return val

        def A2l(m, f):

            k = self.k0(f, 0)
            r = self.r

            v1 = (1j * m * jn(m, r*k)) / r

            return v1

        def A3l(m, f):

            k = self.k0(f, 0)
            r = self.r
            mu = mu0
            lamb = lamb0

            v1 = (2*mu*m*(m + 1)-(r**2)*(k**2)*(lamb + 2*mu))*jn(m, r*k)
            v2 = -2*mu*k*r*jn(m-1, r*k)

            val = (v1+v2) / (r**2)

            return val

        def A4l(m, f):

            k = self.k0(f, 0)
            r = self.r
            mu = mu0

            v1 = -(m+1)*jn(m, r*k)
            v2 = r*k*jn(m-1, r*k)

            val = (2*1j*m*mu) * (v1+v2) / (r**2)

            return val

        def A1t(m, f):

            k = self.k0(f, 1)
            r = self.r

            val = ((1j*m) * jn(m, r*k)) / r

            return val

        def A2t(m, f):

            k = self.k0(f, 1)
            r = self.r

            v1 = -k*r*jn(m-1, r*k)
            v2 = m*jn(m, r*k)

            val = (v1+v2) / r

            return val

        def A3t(m, f):

            k = self.k0(f, 1)
            mu = mu0
            r = self.r

            v1 = -(m+1)*jn(m, r*k)
            v2 = r*k*jn(m-1, r*k)

            val = ((2*1j*m*mu) * (v1+v2)) / (r**2)

            return val

        def A4t(m, f):

            k = self.k0(f, 1)
            mu = mu0
            r = self.r

            v1 = (-2*m*(1 + m) + (k**2)*(r**2))*jn(m, r*k)
            v2 = (2*k*r)*jn(m-1, r*k)

            val = mu*(v1+v2) / (r**2)

            return val

        def B1l(m, f):

            k = self.k0(f, 0)
            r = self.r

            v1 = -m*hn(m, r*k)
            v2 = r*k*hn(m-1, r*k)

            val = (v1+v2) / r

            return val

        def B2l(m, f):

            k = self.k0(f, 0)
            r = self.r

            val = (1j*m) * hn(m, r*k) / r

            return val

        def B3l(m, f):

            k = self.k0(f, 0)
            mu = mu0
            lamb = lamb0
            r = self.r

            v1 = (2*mu*m*(m + 1)-(r**2)*(k**2)*(lamb + 2*mu))*hn(m, r*k)
            v2 = -2*mu*k*r*hn(m-1, r*k)

            val = (v1+v2) / (r**2)

            return val

        def B4l(m, f):

            k = self.k0(f, 0)
            mu = mu0
            r = self.r

            v1 = -(m+1)*hn(m, r*k)
            v2 = r*k*hn(m-1, r*k)

            val = (2*1j*m*mu) * (v1+v2) / (r**2)

            return val

        def B1t(m, f):

            k = self.k0(f, 1)
            mu = mu0
            r = self.r

            val = 1j*m*hn(m, r*k) / r;

            return val

        def B2t(m, f):

            k = self.k0(f, 1)
            r = self.r

            v1 = -k*r*hn(m-1, r*k)
            v2 = m*hn(m, r*k)

            val = (v1+v2) / r

            return val

        def B3t(m, f):

            k = self.k0(f, 1)
            mu = mu0
            r = self.r

            v1 = -(m+1)*hn(m, r*k)
            v2 = r*k*hn(m-1, r*k)

            val = (2*1j*m*mu) * (v1+v2) / (r**2)

            return val

        def B4t(m, f):

            k = self.k0(f, 1)
            mu = mu0
            r = self.r

            v1 = (-2*m*(1 + m) + (k**2)*(r**2))*hn(m, r*k)
            v2 = (2*k*r)*hn(m-1, r*k)

            val = mu*(v1+v2) / (r**2)

            return val

        def C1l(m, f):

            k = self.ks(f, 0)
            r = self.r

            v1 = -m*jn(m, r*k)
            v2 = r*k*jn(m-1, r*k)

            val = (v1+v2) / r

            return val

        def C2l(m, f):

            k = self.ks(f, 0)
            r = self.r

            val = (1j*m) * jn(m, r*k) / r

            return val

        def C3l(m, f):

            k = self.ks(f, 0)
            mu = mus
            lamb = lambs
            r = self.r

            v1 = (2*mu*m*(m + 1)-(r**2)*(k**2)*(lamb + 2*mu))*jn(m, r*k)
            v2 = -2*mu*k*r*jn(m-1, r*k)

            val = (v1+v2) / (r**2)

            return val

        def C4l(m, f):

            k = self.ks(f, 0)
            mu = mus
            lamb = lambs
            r = self.r

            v1 = -(m+1)*jn(m, r*k)
            v2 = r*k*jn(m-1, r*k)

            val = (2*1j*m*mu) * (v1+v2) / (r**2)

            return val

        def C1t(m, f):

            k = self.ks(f, 1)
            r = self.r

            val = (1j*m) * jn(m, r*k) / r

            return val

        def C2t(m, f):

            k = self.ks(f, 1)
            r = self.r

            v1 = -r*k*jn(m-1, r*k)
            v2 = m*jn(m, r*k)

            val = (v1+v2) / r

            return val

        def C3t(m, f):

            k = self.ks(f, 1)
            mu = mus
            r = self.r

            v1 = -(m+1)*jn(m, r*k)
            v2 = r*k*jn(m-1, r*k)

            val = (2*1j*m*mu) * (v1+v2) / (r**2)

            return val

        def C4t(m, f):

            k = self.ks(f, 1)
            mu = mus
            r = self.r

            v1 = (-2*m*(1 + m) + (k**2)*(r**2))*jn(m, r*k)
            v2 = (2*k*r)*jn(m-1, r*k)

            val = mu*(v1+v2) / (r**2)

            return val

        A1 = np.array([[A1l(m, f), A1t(m, f)], [A2l(m, f), A2t(m, f)]])
        B1 = np.array([[B1l(m, f), B1t(m, f)], [B2l(m, f), B2t(m, f)]])
        C1 = np.array([[C1l(m, f), C1t(m, f)], [C2l(m, f), C2t(m, f)]])
        A2 = np.array([[A3l(m, f), A3t(m, f)], [A4l(m, f), A4t(m, f)]])
        B2 = np.array([[B3l(m, f), B3t(m, f)], [B4l(m, f), B4t(m, f)]])
        C2 = np.array([[C3l(m, f), C3t(m, f)], [C4l(m, f), C4t(m, f)]])
        m0 = np.matmul(C2, inv(C1))
        m1 = inv(B2 - np.matmul(m0, B1))
        m2 = np.matmul(m0, A1)
        m3 = m2 - A2

        mat = np.matmul(m1, m3)

        return mat
  
    def G0(self, f, k, pol, cut):

        mat = np.zeros([(2*cut+1), (2*cut+1)], dtype=complex)
        n = 5
        k0_=self.k0(f,pol)
        a = self.a
        lattice = self.lattice

        for i in range(-cut, cut+1):
            for j in range(-cut, cut+1):

                mat[i + cut, j + cut] = S(i, j, k, n, a, k0_, lattice)

        return mat


    def T(self, f, pol1, pol2, cut):

        mat = np.zeros([2*cut+1, 2*cut+1], dtype=complex)

        for i in range(-cut, cut+1):

            mat[i+cut, i+cut] = self.D(i, f)[pol1, pol2]

        return mat

    def G1(self, f, k, cut):

        mat = np.zeros([2*(2*cut+1), 2*(2*cut+1)], dtype=complex)
        SlDll = np.matmul(self.T(f, 0, 0, cut), self.G0(f, k, 0, cut));
        SlDlt = np.matmul(self.T(f, 0, 1, cut), self.G0(f, k, 0, cut));
        StDtl = np.matmul(self.T(f, 1, 0, cut), self.G0(f, k, 1, cut));
        StDtt = np.matmul(self.T(f, 1, 1, cut), self.G0(f, k, 1, cut));
        output = np.block([[SlDll, StDtl],[SlDlt,StDtt]])

        return output - np.identity(2*(2*cut+1))


    def U(self, f, k, cut):
        """
        F(f, k) entrega el determinante al que debemos buscar los ceros
        """
        val = np.real(det(self.G1(f, k, cut)));

        return val

    def V(self, f, k, cut):
        """
        F(f, k) entrega el determinante al que debemos buscar los ceros
        """
        val = np.imag(det(self.G1(f, k, cut)));

        return val

    def Det(self, f, *args):
        "args=(k, cut)"
        k, cut = args
        return [self.U(f, k, cut), self.V(f, k, cut)]


    def solver(self, lista_pruebas, lista_sol, args, max_it = 50):
        i, cut = args
        for prueba in lista_pruebas:
            sol_max = np.max(lista_sol['re'])
            sol = fsolve(self.Det,[prueba,0], args=(i,cut), maxfev=max_it, full_output=True)
            re_w, im_w = sol[0];
            its_sol = sol[2];
            if re_w < 0 or its_sol != 1 or np.isclose(im_w,0)==False:
                pass;
            else:
                if sol_max == 0:
                    lista_sol[0] = (re_w, im_w)
                elif re_w >= lista_pruebas[-1] or repetido(re_w, lista_sol['re']):
                    pass;
                else:
                    for l in range(len(lista_sol['re'])):
                        if lista_sol[l]['re'] == 0:
                            lista_sol[l] = (re_w, im_w)

                            break;
                        elif lista_sol[l]['re'] == sol_max and re_w < sol_max:
                            lista_sol[l] = (re_w, im_w)
                            break;
        return lista_sol


    def zeros(self, prueba0=1.0):
        '''zeros() entrega un arreglo frec de dimensiones n x N x 2, donde n es
        la cantidad de puntos a tomar en el espacio, reciproco, N es la cantidad
        de zeros que queremos que nos entregue por cada valor de k, cut es el
        valor absoluto de los modos que tomara el sistema (ejemplo, si cut = 1,
        el sistema usara los valores de los modos -1, 0 y 1), pace es el avance
        en los ansatz para encontrar las soluciones y prueba0 es el valor
        inicial que tomará en cada punto k'''

        cut = self.cut
        n = self.nk
        N = self.nbands
        pace = self.pace
        k2 = self.k
        tipo=[('re', float),('im', float)];
        its_sol = 0;
        count = 0;
        gamma = False;
        n_prueba = 15;
        self.omega = np.array(np.zeros([N, n]), dtype = tipo)
        print('Initializing...')


        for count,i in enumerate(k2):

            if i == np.pi:
                gamma = True
            else:
                gamma = False

            mat = np.array(np.zeros(N), dtype=tipo)
            if i == 0:
                prueba_min = prueba0
                while True:
                    sol = fsolve(self.Det,[prueba_min,0], args=(i,cut), maxfev=50, full_output=True)
                    re_w, im_w = sol[0];
                    its_sol = sol[2]
                    if re_w < 0 or its_sol != 1 or np.isclose(im_w,0)==False:
                        prueba_min = prueba_min + pace
                    else:
                        break;
                prueba_max = re_w
                lista_pruebas = np.linspace(prueba_min, prueba_max, n_prueba)
            else:
                if gamma:
                    prueba_min = self.omega[1,count-1]['re']*0.8
                    prueba_max = np.max(self.omega[:,count-1]['re'])*1.2
                if np.min(self.omega[-1,count-1]['re']) == 0:
                    prueba_min = prueba0
                    prueba_max = np.min(self.omega[:,count-2]['re'])*1.2
                else:
                    prueba_min = prueba0
                    prueba_max = np.min(self.omega[:,count-1]['re'])*1.2
                lista_pruebas = np.linspace(prueba_min, prueba_max, n_prueba)

            j = 0;
            while j < N:
                lista_pruebas = np.linspace(prueba_min, prueba_max, n_prueba)
                mat = self.solver(lista_pruebas, mat, [i,cut])
                j = n_sol(mat['re'])
                if j < N:
                    prueba_min = prueba_max
                    prueba_max = prueba_min + pace*n_prueba

            mat = np.sort(mat, order='re')
            self.omega[:, count] = mat
            print('k['+str(count)+']')

    def create_folder(self, foldername):

        createFolder(foldername)

    def bandas(self, prueba = 1.0):
        """Modifica el atributo omega, que entrega la relacion de dispersion del
        sistema"""

        self.omega = self.zeros(prueba)
        savefrec(self.k, self.omega, self.foldername)


    def graficar_bandas(self, ylim = 0):
        """Grafica las bandas, con el vector de Bloch en el eje x, frecuencias
        en el eje y, las frecuencias estan normalizadas de la forma
            omega' = omega/(2*pi*Ct0)
        """
        Ct0 = self.vel0[1]

        for i in range(len(self.omega[0,:,0])):
            plt.plot(self.k, self.omega[:,i,0]/(2*np.pi*Ct0), '.', color='black')

        plt.xticks([0, np.pi, 2*np.pi, 3*np.pi], ['X', '$\Gamma$', 'M', 'X'])
        plt.ylabel('$\omega /2\pi C_{t0}$')
        plt.xlabel('k')
        plt.title('Estructura de bandas para ' + str(self.comp) + ' para un radio de llenado x = ' + str(self.filling))
        if ylim !=0:
            plt.ylim(0,ylim)
        plt.savefig(self.frecfolder+'/bandas')
        plt.ylim(0, 2.0)
        plt.savefig(self.frecfolder+'/bandas2')



