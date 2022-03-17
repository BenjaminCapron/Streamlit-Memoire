import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

#VARIABLES
comp_list = []
dict_names = {'L8I3.DE': 'Monétaire','GNR': 'Matières Premières', 'URTH': 'Actions', 'EEM': 'Actions Emergentes', 'AGG': 'Obligations', 'EMHY': 'Obligations Emergentes', 'PEX': 'Private Equity', 'RWR': 'Immobilier'}

def ra(comp_str, comp_list, iteration, weights_ptf, period):
    global returns_ptf, volatilite_ptf, optimal_ptf
    errormsg=0
    try:
        dataframe = yf.download(comp_str, start="2013-06-02", end="2021-06-02")["Adj Close"]
        dataframe = dataframe[comp_list]
        dataframe.rename(columns={'L8I3.DE': 'Monétaire','GNR': 'Matières Premières', 'URTH': 'Actions', 'EEM': 'Actions Emergentes', 'AGG': 'Obligations', 'EMHY': 'Obligations Emergentes', 'PEX': 'Private Equity', 'RWR': 'Immobilier'}, inplace=True)
    except Exception as e:
        st.write("Il faut indiquer à minima 2 classes d'actifs.")
        errormsg=1

    covariance = dataframe.pct_change().apply(lambda x: np.log(1+x)).cov()
    correlation = dataframe.pct_change().apply(lambda x: np.log(1+x)).corr()
    yearly_return = dataframe.resample('Y').last().pct_change().mean()
    annual_std = dataframe.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))

    portfolio_return = []
    portfolio_volatility = []
    portfolio_weights = []

    qt_assets = len(dataframe.columns)
    qt_portfolios = iteration
    my_bar = st.progress(0)
    for portfolio in range(qt_portfolios):
        weights = np.random.random(qt_assets)
        weights = weights/np.sum(weights)
        portfolio_weights.append(weights)
        returns = np.dot(weights, yearly_return) 
        portfolio_return.append(returns)
        var = covariance.mul(weights, axis=0).mul(weights, axis=1).sum().sum()
        std = np.sqrt(var)
        annual_std = std*np.sqrt(250)
        portfolio_volatility.append(annual_std)
        my_bar.progress(portfolio/iteration)
    my_bar.empty()
    
    datas = {'Retours attendus':portfolio_return, 'Volatilité':portfolio_volatility}

    for counter, symbol in enumerate(dataframe.columns.tolist()):
        print(counter, symbol)
        datas[symbol] = [w[counter] for w in portfolio_weights]

    #ACTUAL PTF
    returns_ptf = np.dot(weights_ptf, yearly_return) 
    var_ptf = covariance.mul(weights_ptf, axis=0).mul(weights_ptf, axis=1).sum().sum()
    std_ptf = np.sqrt(var_ptf)
    volatilite_ptf = std_ptf*np.sqrt(250)

    portfolios = pd.DataFrame(datas)
    portfolios.plot.scatter(x='Volatilité', y='Retours attendus', marker='o', s=10, alpha=0.3, grid=True, figsize=[10,10])

    rf = 0.01 # risk factor
    optimal_risky_port = portfolios.iloc[((portfolios['Retours attendus']-rf)/portfolios['Volatilité']).idxmax()]
    plt.scatter(optimal_risky_port[1], optimal_risky_port[0], color='r', marker='x', s=500)

    #ACTUAL PTF
    plt.scatter(volatilite_ptf, returns_ptf, color='y', marker='x', s=500)

    #PTF PROFIL
    optimal_ptf_df = portfolios.loc[portfolios['Volatilité']<(period/100+0.001)]
    optimal_ptf_df = optimal_ptf_df.loc[portfolios['Volatilité']>(period/100-0.001)]
    optimal_ptf_df = optimal_ptf_df.reset_index(drop=True)
    try:
        optimal_ptf = optimal_ptf_df.iloc[optimal_ptf_df['Retours attendus'].idxmax()]
        plt.scatter(optimal_ptf[1], optimal_ptf[0], color='g', marker='x', s=500)
    except:
        optimal_ptf = None

    st.pyplot()
#SideBar
###
with st.sidebar:
    st.title("PROFIL INVESTISSEUR")
    st.subheader("Objectif de volatilité du portefeuille")
    period = st.sidebar.slider("% de votre portefeuille que vous êtes prêt à perdre potentiellement sur une année", min_value=0, max_value=25, value=5, step=1)
    st.subheader("Classes d'actifs à inclure dans le portefeuille")
    monetaire = st.checkbox("Monétaire / Fonds Euros")
    if monetaire:
        comp_list.append("L8I3.DE")
    obligations = st.checkbox("Obligations")
    if obligations:
        comp_list.append("AGG")
    obligations_emergent = st.checkbox("Obligations (Pays Émergents)")
    if obligations_emergent:
        comp_list.append("EMHY")
    actions = st.checkbox("Actions")
    if actions:
        comp_list.append("URTH")
    actions_emergent = st.checkbox("Actions (Pays Émergents)")
    if actions_emergent:
        comp_list.append("EEM")
    immobilier = st.checkbox("Immobilier")
    if immobilier:
        comp_list.append("RWR")
    pe = st.checkbox("Private Equity")
    if pe:
        comp_list.append("PEX")
    mp = st.checkbox("Matières Premières")
    if mp:
        comp_list.append("GNR")
    iteration = st.number_input("Nombre d'itérations", min_value=5000, help="Nombre de portefeuilles différents à simuler (entre 5 000 et 100 000)", step=1000)
    st.title("À propos")
    st.info("""Robo-Advisor réalisé par Benjamin Capron dans le cadre de son PFE.""")
###

#Intro
###
st.title("Allocation Stratégique d'Actifs")
st.subheader("Contexte")
st.write("Cet outil vous permet d'optimiser l'allocation stratégique de votre portefeuille d'actifs. Il vous suffit d'indiquer la composition de votre portefeuille actuel ci-dessous. A noter qu'il faut impérativement indiquer être compétent sur les classes d'actifs qui sont déjà détenues. Le nombre d'itérations permet une plus grande précision mais nécessite en contrepartie un temps de calcul plus élevé.")
###

with st.form(key="option_1"):
    st.subheader("Allocation de votre portefeuille par classe d'actif")
    c1, c2, c3 = st.beta_columns((3,1,3))
    monetaire = c1.number_input("Monétaire / Fonds Euros", min_value=0, help="Indiquez dans ce champ le montant de vos liquidités", step=1000)
    mp = c3.number_input("Matières Premières", min_value=0, help="Indiquez dans ce champ le montant de vos investissement en matières premières", step=1000)
    c1, c2, c3 = st.beta_columns((3,1,3))
    obligations = c1.number_input("Obligations", min_value=0, help="Indiquez dans ce champ le montant de vos obligations (Pays développés)", step=1000)
    obligations_emergent = c3.number_input("Obligations Emergentes", min_value=0, help="Indiquez dans ce champ le montant de vos obligations (Pays émergents)", step=1000)
    actions = c1.number_input("Actions", min_value=0, help="Indiquez dans ce champ le montant de vos actions (Pays développés)", step=1000)
    actions_emergent = c3.number_input("Actions Emergentes", min_value=0, help="Indiquez dans ce champ le montant de vos actions (Pays émergents)", step=1000)
    immobilier = c1.number_input("Immobilier", min_value=0, help="Indiquez dans ce champ le montant de votre investissement en immobilier", step=1000)
    pe = c3.number_input("Private Equity", min_value=0, help="Indiquez dans ce champ le montant de votre investissement en private equity", step=1000)
    submit_button1 = c1.form_submit_button(label="Optimiser le portefeuille")



if submit_button1:
    a1,a2,a3 = st.beta_columns((1,1,1))
    a1.title("")
    a2.title("Votre portefeuille actuel")
    a3.title("")
    st.subheader("")

    c1, c2 = st.beta_columns((1, 1))
    with c1:
        #PTF ACTUEL PLT
        labels = ["Monétaire / Fonds Euros", "Matières Premières", "Obligations", "Obligations Emergentes", "Actions", "Actions Emergentes", "Immobilier", "Private Equity"]
        sizes = [monetaire, mp, obligations, obligations_emergent, actions, actions_emergent, immobilier, pe]
        total = monetaire+mp+obligations+obligations_emergent+actions+actions_emergent+immobilier+pe
        while len(sizes)<8:
            sizes.append(0)
        sizes_dict = {'L8I3.DE':sizes[0], "GNR":sizes[1], "URTH":sizes[4], "EEM":sizes[5], "AGG":sizes[2], "EMHY":sizes[3], "PEX":sizes[7], "RWR":sizes[6]}
        sizes_dict_checker = {k: v for k, v in sizes_dict.items() if v!=0}
        sizes_dict_checker = list(sizes_dict_checker.keys())
        weights_ptf = []
        for comp in comp_list:
            if total==0:
                weights_ptf.append(0)
            else:
                weights_ptf.append(sizes_dict[comp]/total)
        weights_ptf = np.array(weights_ptf)

        #DETIENT COMP QUI NEST PAS EN COMP? - Bug format atm
        i=0
        explode_list = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
        while i<8:
            try:
                if sizes[i]==0:
                    labels.pop(i)
                    sizes.pop(i)
                    explode_list.pop(i)
                    i-=1
            except:
                pass
            i+=1
        colors = ['#ff6666','#66ff66','#66d9ff','#ffff66','#ffb366','#6666ff','#8c66ff', '#d9ff66']
        explode = (explode_list)
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', explode=explode, colors=colors)
        if total!=0:
            st.pyplot(fig1)

        #FT EFFICIENTE
        block=0
        block_list = []
        if total==0:
            st.write("Aucun portefeuille indiqué -> 0% de rentabilité et 0% de volatilité.")
    with c2:
        for element in sizes_dict_checker:
            if element not in comp_list:
                block=1
                block_list.append(element)
        comp_str = ' '.join(comp_list)
        if block==0:
            try:
                ra(comp_str, comp_list, iteration, weights_ptf, period)
            except Exception as e:
                print(e)
                block=1
        else:
            st.write("Vous avez indiqué posséder des classes d'actifs sur lesquelles vous n'êtes pas compétent. Merci de corriger ce point et de relancer l'optimisation :")
            st.write("Compétences manquantes :")
            for missing in block_list:
                st.write(dict_names[missing])

    if block==0:
        with c1:
            if total!=0:
                st.write("Rendements attendus de votre portefeuille : "+str(returns_ptf*100)[0:5]+"% / Volatilité de votre portefeuille : "+str(volatilite_ptf*100)[0:5]+"%.")
        st.write("La croix jaune représente votre portefeuille. La croix verte représente le portefeuille optimal selon votre profil investisseur. La croix rouge représente le portefeuille maximisant les rendements par unité de volatilité (Ratio de Sharpe maximum).")

        st.title("")
        st.title("")
        d1, d2, d3 = st.beta_columns((1, 3, 1))
        st.title("")
        d1.title("")
        d2.title("Le portefeuille optimal selon votre profil")
        d3.title("")
        st.subheader("")

        e1, e2 = st.beta_columns((1, 1))
        with e1:
            st.write("Le portefeuille optimal selon les compétences financières indiquées et une volatilité de "+str(period)+" % présente ces caractéristiques :")
            if optimal_ptf is not None:
                st.write("Les rendements attendus du portefeuille sont estimés à : "+str(optimal_ptf[0]*100)[0:5]+"%.")
                st.write("La volatilité annuelle du portefeuille est estimée à : "+str(optimal_ptf[1]*100)[0:5]+"%.")
                st.subheader("Composition du portefeuille : ")
                optimal_ptf = optimal_ptf.drop(labels=['Retours attendus', 'Volatilité'])
                optimal_ptf = optimal_ptf.rename("Portefeuille optimal")
                st.table(optimal_ptf)
            else:
                st.write("Aucun portefeuille ne peut atteindre une volatilité de "+str(period)+" % en se limitant aux classes d'actifs indiquées dans le profil investisseur.")
        with e2:
            if optimal_ptf is not None:
                fig2, ax2 = plt.subplots()
                explode=[]
                for value in list(optimal_ptf.values):
                    explode.append(0.1)
                ax2.pie(list(optimal_ptf.values), labels=list(optimal_ptf.index), autopct='%1.1f%%', explode=explode, colors=colors)
                st.pyplot(fig2)
            else:
                pass



