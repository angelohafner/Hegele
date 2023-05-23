import streamlit as st
import math
import numpy as np
import pandas as pd
from engineering_notation import EngNumber
import plotly.graph_objects as go
from scipy.optimize import root
import base64
from io import BytesIO
import openpyxl
import os
from pathlib import Path
from scipy.signal import argrelextrema



def f(x0):
    global kappa, epson_0, epson_3, Psi_1, Psi_2, PI
    t1 = epson_0 * epson_3 * kappa**2
    t2 = 2 * Psi_1 * Psi_2
    t3 = Psi_1 * Psi_1 + Psi_2 * Psi_2
    t4 = A_HA / (6 * PI)

    return t1 * (t2 * np.cosh(kappa * x0) - t3) / (2 * np.arcsinh(kappa * x0) * np.arcsinh(kappa * x0)) - t4 / ( x0 * x0 * x0) + A_ST * np.exp(-x0 / h_STR)
def g(x0):
    global t5, t6, t7, Psi_1, Psi_2, kappa, h_STR
    t5 = A_HA / (2 * PI)
    t6 = A_ST / h_STR
    t7 = epson_0 * epson_3 * kappa**2
    return t5 / (x0 * x0 * x0 * x0) - t6 * np.exp(-x0 / h_STR) - (t7 / (np.arcsinh(kappa * x0) * np.arcsinh(kappa * x0) * np.arcsinh(kappa * x0))) * (Psi_2 * np.cosh(kappa * x0) - Psi_1) * (Psi_1 * np.cosh(kappa * x0) - Psi_2)


def limites_eixos(y, x):
    # Calcula a primeira e a segunda derivada de y
    y_prime = np.gradient(y, x)
    indices = np.where((np.abs(y_prime) < 1e20))

    min_critical_value = np.min(y[indices])
    max_critical_value = np.max(y[indices])


    return [min_critical_value, max_critical_value]


def graficos_WP1(xh, PI_ST, PI_VdW, PI_EDL, PI_TOT):
    fig_PI_ST = go.Figure()
    fig_PI_VdW = go.Figure()
    fig_PI_EDL = go.Figure()
    fig_PI_TOT = go.Figure()

    fig_PI_ST.add_trace(go.Scatter(x=xh, y=PI_ST, mode='lines', name='PI_ST'))
    fig_PI_VdW.add_trace(go.Scatter(x=xh, y=PI_VdW, mode='lines', name='PI_VdW'))
    fig_PI_EDL.add_trace(go.Scatter(x=xh, y=PI_EDL, mode='lines', name='PI_EDL'))
    fig_PI_TOT.add_trace(go.Scatter(x=xh, y=(PI_TOT), mode='lines', name='PI_TOT'))

    # lim_inf, lim_sup = limites_eixo_y(PI_ST, xh)
    # fig_PI_ST.update_xaxes(range=[lim_inf, lim_sup])
    # lim_inf, lim_sup = limites_eixo_y(PI_ST, xh)
    # fig_PI_VdW.update_xaxes(range=[lim_inf, lim_sup])
    # lim_inf, lim_sup = limites_eixo_y(PI_ST, xh)
    # fig_PI_EDL.update_xaxes(range=[lim_inf, lim_sup])
    y_lim_inf, y_lim_sup = limites_eixos(PI_TOT, xh)
    fig_PI_TOT.update_yaxes(range=[y_lim_inf, y_lim_sup])




    fig_PI_ST.update_layout(
        title='Structural Forces',
        xaxis_title='xh [m]',
        yaxis_title='F [N/m²]',
    )

    fig_PI_VdW.update_layout(
        title='Van der Walls',
        xaxis_title='xh [m]',
        yaxis_title='F [N/m²]',
    )

    fig_PI_EDL.update_layout(
        title='Electric Double Layer',
        xaxis_title='xh [m]',
        yaxis_title='F [N/m²]',
    )

    fig_PI_TOT.update_layout(
        title='Total',
        xaxis_title='xh [m]',
        yaxis_title='F [N/m²]',
    )

    st.plotly_chart(fig_PI_ST)
    st.plotly_chart(fig_PI_VdW)
    st.plotly_chart(fig_PI_EDL)
    st.plotly_chart(fig_PI_TOT)

def constants():
    global Tk, Pr, F_cal, kappa, A_ST, h_STR, PI, K_b, H_p, F_UV, elec, N_A, epson_0, F_dol, F_qua
    A_ST = 1.5e10  # Structural coefficient (Pa)
    h_STR = 5.0e-11  # Length decay characteristic (m)
    PI = np.pi  # PI number
    K_b = 1.380649e-23  # Boltzmann constant (m 2 Kg s-2 K-1)
    H_p = 6.62607015e-34  # Planck constant (m2 kg s-1)
    F_UV = 3.0e15  # Main electric adsorption frequency (s-1)
    elec = 1.602217663e-19  # Electronic charge (C)
    N_A = 6.02214076e23  # Avogadro number (mol-1)
    epson_0 = 8.854e-12  # Vacuum permittivity (F m-1)


def entrada():
    global Tk, Pr, F_cal, F_dol, F_qua, density_cal, density_dol, density_qua, density_oil, I_ion, IFT, Psi_1, Psi_2, xhmin, xhmax, Prange, erro, N
    col1, col2, col3 = st.columns(3)

    with col1:
        Tk = st.slider('Temperature (K)', min_value=0.1, max_value=1000.0, value=338.0, step=0.10)
        Pr = st.slider('Work Pressure (Pa)', min_value=0.0, max_value=1.0, value=0.01)
        F_cal = st.slider('Average in mass calcite', min_value=0.0, max_value=1.0, value=0.55)
        F_dol = st.slider('Average in mass dolomite', min_value=0.0, max_value=1.0, value=0.38)
        F_qua = st.slider('Average in mass quartz', min_value=0.0, max_value=1.0, value=0.07)

    with col2:
        density_cal = st.slider('Calcite Density (g/cm³)', min_value=1.0, max_value=10.0, value=2.6)
        density_dol = st.slider('Dolomite Density (g/cm³)', min_value=1.0, max_value=10.0, value=1.8)
        density_qua = st.slider('Quartz Density  (g/cm³)', min_value=1.0, max_value=10.0, value=2.65)
        density_oil = st.slider('Oil Density (g/cm³) (15°C & 1 atm)', min_value=0.1, max_value=10.0, value=0.846)

    with col3:
        I_ion = st.slider('Strength ionic (mmol/l)', min_value=0.0, max_value=100., value=10., step=0.1)
        I_ion = 1e-3*I_ion
        IFT = st.slider('Interface Tension (mN/m)', min_value=0.0, max_value=100., value=25.)
        IFT = 1e-3*IFT
        Psi_1 = st.slider('Oil Potential  (mV)', min_value=-100, max_value=100, value=-50)
        Psi_2 = st.slider('Rock Potential (mV)', min_value=-100, max_value=100, value=-50)
        Psi_1 = 1e-3 * Psi_1
        Psi_2 = 1e-3 * Psi_2
        xhmax = st.slider('Maximum thickness (nm)', min_value=0.01, max_value=10.0, value=2.0, step=0.01)
        xhmax = 1e-9*xhmax
        xhmin = xhmax/100
        Prange = int(10000)
        erro = 1e-6
        N = 1000

def piedl(xh, Psi_1, Psi_2, epson_3, kappa):
    temp_1 = epson_0 * epson_3 * (kappa * kappa)
    temp_2 = 2.0 * Psi_1 * Psi_2 * np.cosh(kappa * xh) - ((Psi_1 * Psi_1) + (Psi_2 * Psi_2))
    temp_3 = 2.0 * np.sinh(kappa * xh) * np.sinh(kappa * xh)
    temp_4 = temp_1 * temp_2 / temp_3
    return temp_4

def newton_raphson(x0, N):
    cont = 0
    while True:
        f0 = f(x0)
        g0 = g(x0)

        if (g0 == 0):
            print('Mathematical Error.')
            continua = False
            break

        x1 = x0 - f0 / g0
        x0 = x1
        f1 = f(x1)

        cont += 1
        if cont > N:
            continua = False
            break

        if abs(f1) <= erro:
            continua = True
            break
    return [x1, continua]




def calcular():
    global epson_1, epson_2, epson_3, kappa, n_1, n_2, n_3, A_HA
    epson_3 = 6 * I_ion * (0.08 * I_ion - 1) - 0.165 * (Tk - 273) + 0.019 * Pr + 82.5
    epson_2 = F_cal * (6 * density_cal - 8.1) + F_dol * (1.9 * density_dol) + F_qua * (1.27 * density_qua + 1.25)
    epson_1 = 1.34 + 0.96 * density_oil + 0.015 * math.log(10 * Pr) - 0.00035 * (Tk - 273)
    n_1 = 0.0006 * (Tk - 273) + 0.0004 * Pr + 1.45
    n_2 = 1.724 * F_cal + 1.734 * F_dol + 1.793 * F_qua
    n_temp = 1.34 + 0.13 * Pr - 1.56 * (10 ** -4) * (Tk - 273)
    n_3 = n_temp + I_ion * (2 * (10 ** -1) - 8 * (10 ** -4) * (Tk - 273))
    # -------------------- Calculation constant Hamaker f=0 ----------------------------
    st.write(EngNumber(K_b))
    A_HF0 = K_b * Tk * ((epson_1 - epson_2) / (epson_1 + epson_2)) * ((epson_2 - epson_3) / (epson_2 + epson_3))
    st.write('Hamaker constant | H_h0 = ', str(EngNumber(A_HF0)), 'J/m²')

    # -------------------- Calculation constant Hamaker f>0 ----------------------------
    temp_1 = (3 * H_p * F_UV) / (8 * math.sqrt(2))
    temp_2 = (n_1 * n_1 - n_3 * n_3) * (n_2 * n_2 - n_3 * n_3)
    temp_3 = math.sqrt((n_1 * n_1 + n_3 * n_3) * (n_2 * n_2 + n_3 * n_3))
    temp_4 = (math.sqrt(n_1 * n_1 + n_3 * n_3) + math.sqrt(n_2 * n_2 + n_3 * n_3))
    A_HF1 = temp_1 * (temp_2) / ((temp_3) * (temp_4))
    st.write('Hamaker constant F>0       |      H_h1 = ', A_HF1, ' J')
    # print('--------------------------------------------------------------------------')

    # -------------------- Calculation constant Hamaker A_HA ---------------------------
    A_HA = A_HF0 + A_HF1
    st.write('--------------------------------------------------------------------------')
    st.write('Hamaker constant AH        |     A_HA  = ', A_HA, ' J')
    # print('--------------------------------------------------------------------------')

    # -------------------- Calculation Debye lenght ------------------------------------
    kappa = math.sqrt(((2.0e03 * elec * elec * N_A) / (epson_0 * epson_3 * K_b)) * (I_ion / Tk))
    st.write('--------------------------------------------------------------------------')
    st.write('Debye lenghth              |     kappa = ', kappa, ' m')
    # print('--------------------------------------------------------------------------')

    # -------------------------- calculation thickness equilibrium --------------------


    x0 = xhmin
    f1 = 0
    cont = 1
    continua = True



    x1, continua = newton_raphson(x0, N)

    if continua:
        root = x1
        st.write('Equilibrium thickness      |      h_eq =  ', EngNumber(root), 'm')
        # Estrutural
        xh = np.linspace(xhmin, xhmax, Prange)
        # xh = np.linspace(np.log10(xhmin), np.log10(xhmax), Prange)
        PI_ST = A_ST * np.exp(-xh / h_STR)
        # van der walls
        PI_VdW = -A_HA / (6 * PI * (xh * xh * xh))
        # double layer
        PI_EDL = piedl(xh, Psi_1, Psi_2, epson_3, kappa)
        # total
        PI_TOT = PI_VdW + PI_ST + PI_EDL

        temp5 = epson_0 * epson_3 * kappa * Psi_1 * Psi_2 / IFT
        temp6 = epson_0 * epson_3 * kappa * ((Psi_1 * Psi_1) + (Psi_2 * Psi_2)) / (2 * IFT)
        XEDL = temp5 / np.sinh(kappa * root) - temp6 * np.cosh(kappa * root) / np.sinh(kappa * root)
        XVDW = -A_HA / (12. * PI * IFT * (root * root))
        fit = 932
        XST = fit * A_ST * h_STR * IFT * np.exp(-root / h_STR)
        ARG = 1.0 + XEDL + XVDW + XST
        theta = (180 * np.arccos(ARG)) / PI

        df = pd.DataFrame({
            'Permissividade': [epson_1, epson_2, epson_3],
            'Índice de refração': [n_1, n_2, n_3],
            'Hamaker': EngNumber(A_HF0),
            'theta': EngNumber(theta),
            'XVDW': EngNumber(XVDW),
            'XEDL': EngNumber(XEDL),
        })

    else:
        theta = 'theta error'
        st.write('Not Convergent.')


    return [xh, PI_ST, PI_VdW, PI_EDL, PI_TOT]

def dataframe_to_excel_download_link(xh, PI_ST, PI_VdW, PI_EDL, PI_TOT, filename):
    data = {'xh': xh, 'PI_ST': PI_ST, 'PI_VdW': PI_VdW, 'PI_EDL': PI_EDL, 'PI_TOT': PI_TOT}
    df = pd.DataFrame(data)
    to_write = BytesIO()
    df.to_excel(filename, index=False, engine='openpyxl')

def pag_wp1():
    st.markdown("# WP1")
    entrada()
    constants()
    xh, PI_ST, PI_VdW, PI_EDL, PI_TOT = calcular()
    if st.button('Baixar arquivo Excel'):
        dataframe_to_excel_download_link(xh, PI_ST, PI_VdW, PI_EDL, PI_TOT, 'df_WP1.xlsx')
    graficos_WP1(xh, PI_ST, PI_VdW, PI_EDL, PI_TOT)

def page_mamute():
    st.title("Mamute")
    st.write("Precusa estar logado")

def page_2():
    st.title("Página 2")
    st.write("Esta é a Página 2.")


# Dicionário para mapear funções de página
pages = {
    "Página WP1": pag_wp1,
    "Página Mamute": page_mamute,
    "Página 2": page_2
}



# logged_in = False
# if logged_in:
#     pag_wp1()
# else:
#     logged_in = page_login()


def salvar_usuario(username, password):
    user_data = {'username': [username], 'password': [password]}
    df = pd.DataFrame(user_data)

    if Path('cadastro_usuarios.csv').is_file():
        df_original = pd.read_csv('cadastro_usuarios.csv')
        df_concat = pd.concat([df_original, df], ignore_index=True)
        df_concat.to_csv('cadastro_usuarios.csv', index=False)
    else:
        df.to_csv('cadastro_usuarios.csv', index=False)

def verificar_usuario(username, password):
    if Path('cadastro_usuarios.csv').is_file():
        df = pd.read_csv('cadastro_usuarios.csv')
        usuario_existente = df.loc[df['username'] == username].reset_index(drop=True)

        if not usuario_existente.empty:
            return usuario_existente.at[0, 'password'] == password
    return False

# Inicializa o estado da sessão
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

st.set_page_config(page_title="Login Streamlit", layout="wide")
st.sidebar.title("Autenticação")
login = st.sidebar.radio("Escolha a opção:", ("Login", "Cadastro"))

if login == "Login":
    st.sidebar.header("Login")
    username = st.sidebar.text_input("Usuário")
    password = st.sidebar.text_input("Senha", type='password')

    if st.sidebar.button("Entrar"):
        if verificar_usuario(username, password):
            st.sidebar.success("Login bem-sucedido!")
            st.title("Bem-vindo(a) ao aplicativo!")
            st.session_state.logged_in = True
        else:
            st.sidebar.error("Usuário ou senha incorretos.")
else:
    st.sidebar.header("Cadastro")
    new_username = st.sidebar.text_input("Usuário")
    new_password = st.sidebar.text_input("Senha", type='password')

    if st.sidebar.button("Cadastrar"):
        if new_username and new_password:
            salvar_usuario(new_username, new_password)
            st.sidebar.success("Cadastro realizado com sucesso! Por favor, faça login.")
        else:
            st.sidebar.error("Preencha todos os campos.")

if st.session_state.logged_in:
    pag_wp1()




