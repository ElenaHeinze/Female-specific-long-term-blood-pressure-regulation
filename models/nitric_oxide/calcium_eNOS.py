'''
From Koo, A., Nordsletten, D., Umeton, R., Yankama, B., Ayyadurai, S., García-Cardeña, G., & Dewey, C. F. (2013). In Silico Modeling of Shear-Stress-Induced Nitric Oxide Production in Endothelial Cells through Systems Biology. Biophysical Journal, 104(10), 2295–2306.​

In response to increased fluid shear stress, endothelial cells exhibit a transient increase in cytosolic free calcium (see Fig. 2 A). The influx of calcium
is due to mechanisms such as activation of stress-sensitive calcium channels
and activation of G-protein pathways (6). A calcium channel is directly activated by fluid shear stress, and this leads to intracellular calcium influx.
G-protein-coupled receptors can also be activated by shear stress (7). Activated G-protein induces activity of phospholipase C and production of
inositol 1,4,5-trisphosphate (IP3). IP3 binds to its receptor on the surface
of the endoplasmic reticulum and promotes calcium release from this intracellular storage. The increased intracellular Ca2þ then rapidly binds to
CaM, a calcium-binding protein that significantly upregulates the activity
of eNOS. The elevated intracellular calcium level leads to increased calcium export via the sodium-calcium exchanger and reuptake in intracellular
stores, making increase in intracellular Ca2þ a transient (~5-min) event (8).
To describe the calcium dynamics in response to shear stress, a mathematical model published by Wiesner et al. was used (8,9). This model assumes a step change in calcium influx mediated by the stress-sensitive
calcium channel at the onset of shear stress (10 dynes/cm2).
'''

import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

unisec = 1

# parameters (27 global parameters, unit)
R_T = np.float64(44000) # dimensionless
k1 = np.float64(6e-4)# 1e-9 mol·s^−1
k2 = np.float64(1.000) # s^−1
k3 = np.float64(3.320) # s^−1
k4 = np.float64(2500.000)# 10−9 mol·s^−1
k5 = np.float64(5e-11)# 10−9 mol^−1 ·s^−1
k6 = np.float64(0.050)# 10−9 mol^−1 ·s^−1
k7 = np.float64(150.000) # s−1 
K1 = np.float64(0.000)# 10−9 mol 
K2 = np.float64(200.000)# 10−9 mol 
K3 = np.float64(150.000)# 10−9 mol
K4 = np.float64(80.000)# 10−9 mol
K5 = np.float64(321.000)# 10−9 mol
K_hi = np.float64(380.000)# 10−9 mol
k_CICR = 1.000 # dimensionless
K_CICR = 0.000# 10−9 mol
k_CCE = 0.000# 10−9 mol−1·s−1
B_T = np.float64(120000.000)# 10−9 mol
dot_Vp = np.float64(815.000)# 10−9 mol·s−1
dot_Vex = np.float64(9165.000)# 10−9 mol·s−1 
dot_Vhi = np.float64(2380.000)# 10−9 mol·s−1
dot_q_inpass = np.float64(3000.000)  # 10−9 mol·s−1
Cao = np.float64(100.000)# 10−9 mol
tau_I = np.float64(66.000)# 10−9 mol
tau_II = np.float64(0.010)# 10−9 mol
half  = 0.500 # dimensionless
fracK = np.float64(7071067.810)# 10−9 mol

# function that returns dy/dt
def model(t, y:list):
    Ca_ex, Ca_s, Ca_c, Ca_B, s5, ip3, s7, s8, s9, s10, s11, TimeT, s13, ShearStress = y
    
    # reactions
    re3 = -k6 * Ca_c * (B_T - Ca_B) + k7 * Ca_B
    re4 = k_CCE * (((fracK*Cao)/(K3+Cao)) - Ca_s) * (Ca_ex-Ca_s) 
    re5 = (k1 * (R_T - half*R_T*(np.exp(-TimeT/tau_I) + np.exp(-TimeT/tau_II) + ((np.exp(-TimeT/tau_I)-np.exp(-TimeT/tau_II)) * (tau_I+tau_II))/(tau_I-tau_II))))*Ca_c/(K1 + Ca_c)
    re6 = (k2 * ip3) 
    re7 = k3 * (k_CICR * Ca_c)/(K_CICR + Ca_c) * ((ip3/(K2 + ip3))**3) * Ca_s - k4 * ((Ca_c/(K3 + Ca_c))**2) + k5 *Ca_s*Ca_s
    re8 = (dot_Vhi * (Ca_c**4))/((K_hi**4) + (Ca_c**4))
    re9 = (dot_Vex * Ca_c)/(K5 + Ca_c)
    re10 = dot_q_inpass
    re11 = unisec
    re12 = (dot_Vp * (Ca_c**2))/(K4**2 + Ca_c**2)

    # differential equations
    dCa_exdt = 0
    dCa_sdt = re4 - re7
    dCa_cdt = re3 + re7 - re8 - re9 + re10 - re12
    dCa_Bdt = - re3
    ds5dt = - re4
    dip3dt = re5 - re6
    ds7dt = - re5
    ds8dt = re6
    ds9dt = re8 + re12
    ds10dt = re9
    ds11dt = - re10
    dTimeTdt = re11
    ds13dt = - re11
    dtShearStressdt = 0

    return [dCa_exdt, dCa_sdt, dCa_cdt, dCa_Bdt, ds5dt, dip3dt, ds7dt, ds8dt, ds9dt, ds10dt, ds11dt, dTimeTdt, ds13dt, dtShearStressdt]

# initial conditions taken from SBML model report (May 5, 2016)
# species (14 species in total)
# [Ca_ex, Ca_s, Ca_c, Ca_B, s5, IP3, s7, s8, s9, s10, s11, TimeT, s13, ShearStress]
y0 = [np.float64(1500000), np.float64(2830000), np.float64(117.2), np.float64(3870), 0,0,0,0,0,0,0,0,0,0]

# time points
sec = 1000
t = [0,sec]
L = len(t)

# solve ODEs
# mode run
df = pd.read_csv("./Female-specific-long-term-blood-pressure-regulation/COPASI/calcium_eNOS/Koo.txt", delimiter="\t")
t_eval = df['# Time'].to_numpy() # np.linspace(0, sec, 1000)
print("t_eval:", t_eval)
sol = solve_ivp(model,t, y0, t_eval=t_eval, method='LSODA', rtol=1e-6,atol=1e-12)

# Plot the time course of the solution
print("length of sol.t:", len(sol.t))
#plt.plot(sol.t, sol.y[0, :], 'r-', label='Ca_ex')
plt.plot(sol.t, sol.y[1, :], 'b-', label='Ca_s')
plt.plot(t_eval, df['Ca_s'], 'silver', linestyle='dashdot', label='Ca_s Koo et al.')
plt.plot(sol.t, sol.y[2, :], 'g-', label='Ca_c')
plt.plot(t_eval, df['Ca_c'], 'silver', linestyle='dashdot', label='Ca_c Koo et al.')
plt.plot(sol.t, sol.y[3, :], 'c-', label='Ca_B')
plt.plot(t_eval, df['Ca_B'], 'silver', linestyle='dashdot', label='Ca_B Koo et al.')
#plt.plot(sol.t, sol.y[4, :], 'k-', label='s5')
plt.plot(sol.t, sol.y[5, :], 'y-', label='IP3')
plt.plot(t_eval, df['IP3'], 'silver', linestyle='dashdot', label='IP3 Koo et al.')
#plt.plot(sol.t, sol.y[6, :], 'k-', label='s7')
plt.plot(sol.t, sol.y[7, :], 'mediumpurple', label='s8')
plt.plot(t_eval, df['s8'], 'silver', linestyle='dashdot', label='s8 Koo et al.')
plt.plot(sol.t, sol.y[8, :], 'orange', label='s9')
plt.plot(t_eval, df['s9'], 'silver', linestyle='dashdot', label='s9 Koo et al.')
plt.plot(sol.t, sol.y[9, :], 'teal', label='s10')
plt.plot(t_eval, df['s10'], 'silver', linestyle='dashdot', label='s10 Koo et al.')
#plt.plot(sol.t, sol.y[10, :], 'r', label='s11')
#plt.plot(sol.t, sol.y[12, :], 'springgreen', label='s13')

plt.grid()
plt.ylim(0, 5e6)
plt.legend(loc='best')
plt.xlabel('time (sec)')
plt.ylabel('concentration nmol/L')
plt.title('Shear stress induced calcium influx and eNOS activation')
plt.show()

# calculate the mean squared error and relative deviation to investigate implementation performance
mse_Ca_c = (np.square(sol.y[2, :] - df['Ca_c'])).mean(axis=None)
relative_deviation_Ca_c = np.mean(np.abs((sol.y[2, :] - df['Ca_c']) / df['Ca_c']))
mse_Ca_B = (np.square(sol.y[3, :] - df['Ca_B'])).mean(axis=None)
relative_deviation_Ca_B = np.mean(np.abs((sol.y[3, :] - df['Ca_B']) / df['Ca_B']))
mse_IP3 = (np.square(sol.y[5, :] - df['IP3'])).mean(axis=None)
relative_deviation_IP3 = np.mean(np.abs((sol.y[5, :] - df['IP3']) / df['IP3']))
mse_s8 = (np.square(sol.y[7, :] - df['s8'])).mean(axis=None)
relative_deviation_s8 = np.mean(np.abs((sol.y[7, :] - df['s8']) / df['s8']))
mse_s9 = (np.square(sol.y[8, :] - df['s9'])).mean(axis=None)
relative_deviation_s9 = np.mean(np.abs((sol.y[8, :] - df['s9']) / df['s9']))
mse_s10 = (np.square(sol.y[9, :] - df['s10'])).mean(axis=None)
relative_deviation_s10 = np.mean(np.abs((sol.y[9, :] - df['s10']) / df['s10']))

print("MSE Ca_c:", mse_Ca_c)
print("Relative deviation Ca_c:", relative_deviation_Ca_c)
print("MSE Ca_B:", mse_Ca_B)
print("Relative deviation Ca_B:", relative_deviation_Ca_B)
print("MSE IP3:", mse_IP3)
print("Relative deviation IP3:", relative_deviation_IP3)
print("MSE s8:", mse_s8)
print("Relative deviation s8:", relative_deviation_s8)
print("MSE s9:", mse_s9)
print("Relative deviation s9:", relative_deviation_s9)
print("MSE s10:", mse_s10)
print("Relative deviation s10:", relative_deviation_s10)

plt.plot(sol.t, (np.square(sol.y[2, :] - df['Ca_c'])), 'g-', label='Ca_c')
plt.plot(sol.t, (np.square(sol.y[3, :] - df['Ca_B'])), 'c-', label='Ca_B')
plt.plot(sol.t, (np.square(sol.y[5, :] - df['IP3'])), 'y-', label='IP3')
plt.plot(sol.t, (np.square(sol.y[7, :] - df['s8'])), 'mediumpurple', label='s8')
plt.plot(sol.t, (np.square(sol.y[8, :] - df['s9'])), 'orange', label='s9')
plt.plot(sol.t, (np.square(sol.y[9, :] - df['s10'])), 'teal', label='s10')
plt.legend(loc='best')
plt.xlabel('time (sec)')
plt.ylabel('sqared error')
plt.title('Squared error of the model implementation')
plt.show()
