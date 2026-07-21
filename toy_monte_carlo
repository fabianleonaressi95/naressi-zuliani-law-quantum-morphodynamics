import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Impostazione Seed per la riproducibilità (Capitolo 12.10)
np.random.seed(42)

# ==========================================
# 1. PARAMETRI DELLA SIMULAZIONE
# ==========================================
N_EVENTS = 5000       # Numero di eventi per classe
N_JETS = 6            # Numero di jet per evento
GAMMA = 50.0          # Parametro di bilanciamento dello Score (Capitolo 11.9)
N_MODES = 8           # Numero di modi della Trasformata Aperiodica (n = 1..N_MODES)

print(f"Generazione di {N_EVENTS} eventi per Segnale (WW VBS) e Fondo (W/Z+jets)...")

# ==========================================
# 2. GENERAZIONE EVENTI TOY MONTE CARLO (12.3)
# ==========================================
def generate_events(n_events, n_jets, is_signal=True):
    """
    Genera cinematica dei jet (pT, eta).
    - Segnale (VBS): jet più separati in rapidità (eta elevata) e pT elevato.
    - Fondo (QCD): jet più centrali e pT a decadimento esponenziale.
    """
    events = []
    for _ in range(n_events):
        if is_signal:
            # Segnale VBS: due jet di tagging ad alta eta + jet centrali
            eta_tag = np.random.choice([-1, 1], size=2) * np.random.normal(location=3.5, scale=0.8, size=2)
            eta_extra = np.random.normal(location=0.0, scale=1.2, size=n_jets - 2)
            eta = np.concatenate([eta_tag, eta_extra])
            pt = np.random.gamma(shape=3.0, scale=40.0, size=n_jets) + 20.0
        else:
            # Fondo QCD: distribuzione stretta attorno a eta=0, pT più morbido
            eta = np.random.normal(location=0.0, scale=1.8, size=n_jets)
            pt = np.random.exponential(scale=35.0, size=n_jets) + 20.0
            
        events.append({'pt': pt, 'eta': eta})
    return events

signal_events = generate_events(N_EVENTS, N_JETS, is_signal=True)
background_events = generate_events(N_EVENTS, N_JETS, is_signal=False)

# ==========================================
# 3. TRASFORMATA APERIODICA & METRICHE (12.4 - 12.6)
# ==========================================
def compute_metrics(event):
    """
    Calcola A(n), normalizza il vettore psi, costruisce la matrice di densità
    e ricava IPR, Entropia vN e Score.
    """
    pt = event['pt']
    eta = event['eta']
    
    # 3.1 Coefficienti A(n) con Kernel di Bessel Jn(eta_i) * pT_i
    A = np.zeros(N_MODES)
    for n in range(1, N_MODES + 1):
        A[n-1] = np.sum(sp.jn(n, eta) * pt)
    
    # 3.2 Normalizzazione per lo stato psi nello spazio di Hilbert (12.5)
    norm = np.linalg.norm(A)
    if norm == 0:
        psi = np.ones(N_MODES) / np.sqrt(N_MODES)
    else:
        psi = A / norm
        
    # 3.3 Inverse Participation Ratio (IPR)
    ipr = np.sum(psi**4)
    
    # 3.4 Matrice di densità rho = |psi><psi| e autovalori per Entropia vN
    rho = np.outer(psi, psi)
    eigenvalues = np.linalg.eigvalsh(rho)
    # Filtra autovalori strettamente positivi per evitare log(0)
    pos_evals = eigenvalues[eigenvalues > 1e-12]
    s_vn = -np.sum(pos_evals * np.log(pos_evals))
    
    # 3.5 Score Informazionale (11.9)
    score = ipr - GAMMA * s_vn
    
    return ipr, s_vn, score

def process_dataset(events):
    iprs, svns, scores = [], [], []
    for ev in events:
        ipr, svn, sc = compute_metrics(ev)
        iprs.append(ipr)
        svns.append(svn)
        scores.append(sc)
    return np.array(iprs), np.array(svns), np.array(scores)

# Calcolo osservabili per segnale e fondo
ipr_sig, svn_sig, score_sig = process_dataset(signal_events)
ipr_bkg, svn_bkg, score_bkg = process_dataset(background_events)

# ==========================================
# 4. ROC CURVE & AUC (12.9)
# ==========================================
y_true = np.concatenate([np.ones(N_EVENTS), np.zeros(N_EVENTS)])
y_scores = np.concatenate([score_sig, score_bkg])

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

print(f"\n--- RISULTATI ANALISI TOY MONTE CARLO ---")
print(f"IPR Medio  -> Segnale: {np.mean(ipr_sig):.4f} | Fondo: {np.mean(ipr_bkg):.4f}")
print(f"S_vN Media -> Segnale: {np.mean(svn_sig):.4f} | Fondo: {np.mean(svn_bkg):.4f}")
print(f"Score Medio -> Segnale: {np.mean(score_sig):.4f} | Fondo: {np.mean(score_bkg):.4f}")
print(f"AUC Calcolata: {roc_auc:.4f}")

# ==========================================
# 5. VISUALIZZAZIONE GRAFICA
# ==========================================
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Grafico 1: Distribuzione IPR
axs[0].hist(ipr_sig, bins=40, alpha=0.6, label='WW VBS (Segnale)', color='blue', density=True)
axs[0].hist(ipr_bkg, bins=40, alpha=0.6, label='W/Z+jets (Fondo)', color='red', density=True)
axs[0].set_title('Inverse Participation Ratio (IPR)')
axs[0].set_xlabel('IPR')
axs[0].set_ylabel('Densità')
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# Grafico 2: Distribuzione Score Informazionale
axs[1].hist(score_sig, bins=40, alpha=0.6, label='WW VBS (Segnale)', color='blue', density=True)
axs[1].hist(score_bkg, bins=40, alpha=0.6, label='W/Z+jets (Fondo)', color='red', density=True)
axs[1].set_title(f'Score Informazionale (γ = {GAMMA})')
axs[1].set_xlabel('Score')
axs[1].set_ylabel('Densità')
axs[1].legend()
axs[1].grid(True, alpha=0.3)

# Grafico 3: Curva ROC
axs[2].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
axs[2].plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
axs[2].set_xlim([0.0, 1.0])
axs[2].set_ylim([0.0, 1.05])
axs[2].set_xlabel('False Positive Rate')
axs[2].set_ylabel('True Positive Rate')
axs[2].set_title('Curva ROC - Capacità Discriminante')
axs[2].legend(loc="lower right")
axs[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
