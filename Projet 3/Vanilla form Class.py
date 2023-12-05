import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

class MonteCarlo:
    """Class for Monte Carlo simulation of a vanilla option"""

    def __init__(self, S_0, K, sigma, mu, Ns, maturity, ic):
        """
        Initialize MonteCarlo object

        Args:
            S_0 (float): Initial stock price
            K (float): Strike price
            sigma (float): Volatility
            mu (float): Drift
            Ns (int): Number of simulations
            maturity (int): Maturity of the option in years
            ic (float): Confidence interval
        """
        self.S_0 = S_0
        self.K = K
        self.sigma = sigma
        self.mu = mu
        self.Ns = Ns
        self.maturity = maturity
        self.ic = ic

    def generate_t(self):
        """Generate time vector

        Returns:
            List : Time vector
        """
        t_1 = np.arange(0, 60/252, 1/252)
        t_2 = np.arange(self.maturity-59/252, self.maturity, 1/252)
        return np.concatenate((t_1, t_2), axis=0)

    def simulate_gbm(self, t):
        """Simulate a GBM trajectory

        Returns:
            List : A GBM trajectory
        """
        S = [self.S_0]
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            S.append( S[-1] * np.exp( (self.mu-(self.sigma**2)/2)*dt + self.sigma*np.sqrt(dt)*np.random.normal(0, 1) ) )
        return S

    def payoff(self, S):
        """Calculate the payoff of a trajectory

        Args:
            S (List): Trajectory

        Returns:
            _type_: Payoff of the asset
        """
        return np.max([np.mean(S[61:])/np.mean(S[:60]) - self.K, 0])

    def simulate_monte_carlo(self):
        """Simulate Ns trajectories and calculate the payoff of each

        Returns:
            List: The payoff of each trajectory
        """
        t = self.generate_t()
        P = []
        for i in range(self.Ns):
            S = self.simulate_gbm(t) 
            P.append(self.payoff(S))
        return P

    def convergence_mc(self, P):
        """Perform a Monte Carlo method

        Returns:
            M: Vector of the moving average
            b_inf: Vector of the lower bound
            b_sup: Vector of the upper bound
        """
        a = norm.ppf(self.ic)
        M = []
        ET = []
        b_inf = []
        b_sup = []
        for i in range(len(P)):
            M.append(np.mean(P[:i+1]))
            ET.append(np.std(P[:i+1]))
            b_inf.append( M[-1] - a*ET[-1]/np.sqrt(i) )
            b_sup.append( M[-1] + a*ET[-1]/np.sqrt(i) )
        return M, b_inf, b_sup

    def generate_convergence_curve(self):
        """Generate a convergence curve plot

        Returns:
            Figure: The convergence curve plot
        """
        P = self.simulate_monte_carlo()
        M, b_inf, b_sup = self.convergence_mc(P)
        fig = Figure(figsize=(10, 4))
        plot = fig.add_subplot(111)
        plot.plot(range(self.Ns-1), M[1:], 'g', label = "Moyenne")
        plot.plot(range(self.Ns-1), b_inf[1:], 'r', label = "Borne inférieure")
        plot.plot(range(self.Ns-1), b_sup[1:], 'r', label = "Borne supérieure")
        plot.set_xlabel("Nombre de simulations")
        plot.set_ylabel("Prix")
        final_price = M[-1]
        error = b_sup[-1] - b_inf[-1]
        plot.set_title(f"Courbe d'évolution du prix en fonction du nombre de simulation de Monte Carlo\nFinal price: {final_price:.4f}, Error: {error:.4f}")
        plot.grid()
        plot.legend()
        return fig

class Form:
    """Class for GUI form"""

    def __init__(self):
        """Initialize Form object"""
        self.form = tk.Tk()
        self.form.title("Monte Carlo Convergence Curve")
        self.form.geometry("800x600")
        self.canvas = None
        self.toolbar = None

        self.S_0_label = tk.Label(self.form, text="S_0:")
        self.S_0_label.pack()
        self.S_0_entry = tk.Entry(self.form)
        self.S_0_entry.pack()

        self.K_label = tk.Label(self.form, text="K:")
        self.K_label.pack()
        self.K_entry = tk.Entry(self.form)
        self.K_entry.pack()

        self.sigma_label = tk.Label(self.form, text="sigma:")
        self.sigma_label.pack()
        self.sigma_entry = tk.Entry(self.form)
        self.sigma_entry.pack()

        self.mu_label = tk.Label(self.form, text="mu:")
        self.mu_label.pack()
        self.mu_entry = tk.Entry(self.form)
        self.mu_entry.pack()

        self.maturity_label = tk.Label(self.form, text="Maturity (in years):")
        self.maturity_label.pack()
        self.maturity_entry = tk.Entry(self.form)
        self.maturity_entry.pack()

        self.Ns_label = tk.Label(self.form, text="Ns:")
        self.Ns_label.pack()
        self.Ns_entry = tk.Entry(self.form)
        self.Ns_entry.pack()

        self.ic_label = tk.Label(self.form, text="ic:")
        self.ic_label.pack()
        self.ic_entry = tk.Entry(self.form)
        self.ic_entry.pack()

        self.generate_button = tk.Button(self.form, text="Generate Convergence Curve", command=self.display_convergence_curve)
        self.generate_button.pack()

        self.reset_button = tk.Button(self.form, text="Reset Form", command=self.reset_form)
        self.reset_button.pack()

    def display_convergence_curve(self):
        """Display the convergence curve plot"""
        S_0 = float(self.S_0_entry.get())
        K = float(self.K_entry.get())
        sigma = float(self.sigma_entry.get())
        mu = float(self.mu_entry.get())
        Ns = int(self.Ns_entry.get())
        ic = float(self.ic_entry.get())
        maturity = int(self.maturity_entry.get())

        monte_carlo = MonteCarlo(S_0, K, sigma, mu, Ns, maturity, ic)

        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()

        fig = monte_carlo.generate_convergence_curve()
        self.canvas = FigureCanvasTkAgg(fig, master=self.form)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.form)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def reset_form(self):
        """Reset the form"""
        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()
            self.toolbar.pack_forget()

    def run(self):
        """Run the form"""
        self.form.mainloop()

form = Form()
form.run()