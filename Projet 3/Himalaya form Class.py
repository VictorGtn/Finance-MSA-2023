import numpy as np
from scipy.stats import norm
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

class MonteCarlo:
    def __init__(self, S_0, K, sigma, mu, Ns, n,ic):
        """Initializes all the parameters of the Monte Carlo simulation.

        Args:
            S_0 (List): Starting values of our assets.
            K (int): Strike price.
            sigma (int): Variance of our diffusion model.
            mu (int): Mean of our diffusion model.
            Ns (int): Number of simulations.
            n (int): Number of assets.
            ic (float): Confidence level (between 0 and 1).
        """
        self.S_0 = S_0
        self.K = K
        self.sigma = sigma
        self.mu = mu
        self.Ns = Ns
        self.n = n
        self.ic = ic

    def generate_t(self):
        """Generates the time vector.

        Returns:
            List : Time vector.
        """
        t = [i for i in range(self.n)]
        return t

    def simulate_gbm(self,t):
        """Simulates a trajectory.

        Returns:
            List : A trajectory.
        """
        Sn=[]
        for i in range(len(self.S_0)):
            S = [self.S_0[i]]
            for i in range(1, len(t)):
                dt = t[i] - t[i-1]
                S.append( S[-1] * np.exp( (self.mu-(self.sigma**2)/2)*dt + self.sigma*np.sqrt(dt)*np.random.normal(0, 1) ) )
            Sn.append(S)
        return Sn

    def payoff(self, S):
        """Calculates the payoff of our trajectory.

        Args:
            S (List): Trajectory.
            K (_type_): Strike price.

        Returns:
            _type_: Payoff of our asset.
        """
        payoff=0
        i=len(S)
        n=len(S)
        while i>1:
            R = [S[j][n-i+1]/S[j][0] for j in range(len(S))]
            i-=1
            argmax = R.index(max(R))
            payoff += (R[argmax]-self.K)
            del S[argmax]
        return max(payoff,0)

    def simulate_monte_carlo(self):
        """Simulates Ns trajectories and calculates the payoff of each of them.

        Returns:
            List: The payoff of each of our trajectories.
        """
        t = self.generate_t()
        P = []
        for i in range(self.Ns):
            S = self.simulate_gbm(t)
            P.append(self.payoff(S))
        return P

    def convergence_mc(self, P):
        """Performs a Monte Carlo method.

        Args:
            ic (int): Confidence level.

        Returns:
            M: Vector of the moving average.
            b_inf: Vector of the lower bound.
            b_sup: Vector of the upper bound.
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
        """Generates the Monte Carlo convergence curve.

        Returns:
            Figure: The convergence curve figure.
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
    """Class that creates the form to enter the parameters of the option.
    """
    def __init__(self):
        """Creates the form to enter the parameters of the option.
        """
        self.form = tk.Tk()
        self.form.title("Monte Carlo Convergence Curve")
        self.form.geometry("800x600")
        self.canvas = None
        self.toolbar = None


        self.S0_label = tk.Label(self.form, text="S0 (comma-separated list):")
        self.S0_label.pack()
        self.S0_entry = tk.Entry(self.form)
        self.S0_entry.pack()

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

        self.n_label = tk.Label(self.form, text="Number of asset (thus of maturity):")
        self.n_label.pack()
        self.n_entry = tk.Entry(self.form)
        self.n_entry.pack()

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

    def reset_form(self):
        """Clears the current graph.
        """
        self.canvas.get_tk_widget().pack_forget()
        self.toolbar.pack_forget()

    def display_convergence_curve(self):
        """Generates the Monte Carlo convergence curve graph.
        """
        S0_str = self.S0_entry.get()
        S0_values = [float(s) for s in S0_str.split(',')]
        K = float(self.K_entry.get())
        sigma = float(self.sigma_entry.get())
        mu = float(self.mu_entry.get())
        Ns = int(self.Ns_entry.get())
        ic = float(self.ic_entry.get())
        n = int(self.n_entry.get())

        mc = MonteCarlo(S0_values, K, sigma, mu, Ns, n, ic)
        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()

        fig = mc.generate_convergence_curve()
        self.canvas = FigureCanvasTkAgg(fig, master=self.form)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.form)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
    
    def reset_form(self):
        """Clears the current graph.
        """
        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()
            self.toolbar.pack_forget()


    def run(self):
        """Runs the form.
        """
        self.form.mainloop()

form = Form()
form.run()